import importlib
import os
from abc import ABCMeta, abstractmethod
from dataclasses import asdict, dataclass, field
from warnings import warn

import click
from aiida import load_profile
from aiida.engine import if_, submit, WorkChain
from ase.io.formats import UnknownFileTypeError
from tqdm import tqdm

from ecint.config import RESULT_NAME
from ecint.postprocessor.utils import notification_in_dingtalk
from ecint.preprocessor.kind import KindSection
from ecint.preprocessor.utils import load_config, load_kind, \
    load_machine, load_structure

load_profile()


# @dataclass
# class SubData(object):
#     config: str or dict = None
#     kind_section: str or dict or list = None
#     machine: str or dict = None


@dataclass
class BaseUserInput(object, metaclass=ABCMeta):
    workflow: str
    webhook: str = None
    resdir: str = field(default=os.getcwd())
    # metadata: SubData and other special paras
    metadata: dict = field(default_factory=dict)
    # subdata: {str: SubData(config, kind_section, machine), ...}
    subdata: dict = field(default_factory=dict)

    @abstractmethod
    def get_workflow_inp(self):
        pass


@dataclass
class DpUserInput(BaseUserInput):
    datadirs: list = None

    def get_workflow_inp(self):
        workflow_inp = {'datadirs': list(map(os.path.abspath, self.datadirs)),
                        **load_input(asdict(self), resdir=self.resdir)}
        return workflow_inp


@dataclass
class SmUserInput(BaseUserInput):
    # structure section
    # if set, high-throughput calc will run, conflicting with `structure`
    structures_folder: str = None
    is_batch: bool = False
    structure: str or list = None
    format: str = None
    cell: list = field(default_factory=list)
    pbc: bool or list = True
    masses: dict = None
    options: dict = field(default_factory=dict)

    @property
    def has_structures_folder(self):
        if self.structure and self.structures_folder:
            raise KeyError('`structure` and `structures_folder` '
                           'can not coexist')
        return True if self.structures_folder else False

    def get_workflow_inp(self):
        if self.has_structures_folder:
            if os.path.isdir(self.structures_folder):
                structure_settings = {
                    'cell': self.cell,
                    'pbc': self.pbc,
                    'masses': self.masses,
                    'format': self.format,
                    **self.options
                }
                structures = _load_sfolder(self.structures_folder,
                                           **structure_settings)
                if not self.is_batch:
                    workflow_inp = []
                    for i, structure in enumerate(structures):
                        resdir = os.path.join(self.resdir, str(i))
                        workflow_inp.append({'structure': structure,
                                             **load_input(asdict(self), resdir)}
                                            )
                else:
                    workflow_inp = {'structures': structures,
                                    **load_input(asdict(self), self.resdir)}
            else:
                raise ValueError('`structures_folder` is not a folder')
        else:
            workflow_inp = {**load_s(asdict(self)),
                            **load_input(asdict(self), resdir=self.resdir)}
        return workflow_inp


@dataclass
class MixUserInput(BaseUserInput):
    datadirs: list = None
    # structure related, structures_folder and variables in lmp.in
    imd: list = None
    # format: str = None
    # cell: list = field(default_factory=list)
    # pbc: bool or list = True
    # masses: dict = None
    options: dict = field(default_factory=dict)
    kinds: list = None
    descriptor_sel: list = None

    def get_workflow_inp(self):
        # convert structures_folder in imd
        imd = []
        for setting in self.imd:
            structures = _load_sfolder(setting.pop('structures_folder'),
                                       **self.options)
            # for structure in structures:
            #     structure.store()
            imd.append({'structures': structures, **setting})
        workflow_inp = {
            'datadirs': list(map(os.path.abspath, self.datadirs)),
            'imd': imd,
            'kinds': self.kinds,
            'descriptor_sel': self.descriptor_sel,
            **load_input(asdict(self), resdir=self.resdir)
        }
        return workflow_inp


def create_userinput(workflow_name):
    workflow = load_workflow(workflow_name)
    workflow_type = workflow.TYPE
    if workflow_type == 'simulation':
        return SmUserInput
    elif workflow_type == 'deepmd':
        return DpUserInput
    elif workflow_type == 'mixing':
        return MixUserInput


# TODO: use this method temporarily for testing, when users input,
#  please use absolutely path directly
def convert_graphs_path(graphs):
    if isinstance(graphs, dict):
        for k, v in graphs.items():
            graphs[k] = list(map(os.path.abspath, v))
        return graphs
    elif isinstance(graphs, list):
        return list(map(os.path.abspath, graphs))
    else:
        raise TypeError('Graphs need be dict or list')


def load_workflow(workflow_name):
    """Convert warkflow name to WorkChain Process

    Args:
        workflow_name (str): name of user input workflow

    Returns:
        WorkChain: workflow in ecint

    """
    workflow_lib = importlib.import_module('ecint.workflow')
    workflow = getattr(workflow_lib, workflow_name)
    return workflow


def _load_subdata(subdata):
    workflow_inp = {}
    # check config
    if subdata.get('config'):
        config = subdata.pop('config')
        workflow_inp.update({'config': load_config(config)})
    # check kind_section
    if subdata.get('kind_section'):
        kind_section = subdata.pop('kind_section')
        if (isinstance(kind_section, dict) and
                ('BASIS_SET' in kind_section) and
                ('POTENTIAL' in kind_section)):
            _kind_section = KindSection(kind_section['BASIS_SET'],
                                        kind_section['POTENTIAL'])
            workflow_inp.update({'kind_section': _kind_section})
        else:
            workflow_inp.update({'kind_section': load_kind(kind_section)})
    # check machine
    if subdata.get('machine'):
        machine = subdata.pop('machine')
        workflow_inp.update({'machine': load_machine(machine)})
    # TODO: remove when remove get_abs_path
    if subdata.get('graphs'):
        graphs = convert_graphs_path(subdata.pop('graphs'))
        workflow_inp.update({'graphs': graphs})
    if subdata.get('template'):
        template = os.path.abspath(subdata.pop('template'))
        workflow_inp.update({'template': template})
    # other params
    workflow_inp.update(**subdata)
    return workflow_inp


def _load_metadata(metadata):
    return {**_load_subdata(metadata)}, {**metadata}


def _load_sfolder(structures_folder, **structure_kwargs):
    print('Convert Structures...')
    structure_bar = tqdm(os.listdir(structures_folder))
    structures = []
    for structure_file in structure_bar:
        try:
            structure_bar.set_description(f'Upload {structure_file}')
            structure_dir = os.path.join(structures_folder, structure_file)
            structures.append(load_structure(structure_dir, **structure_kwargs))
        except UnknownFileTypeError as te:
            warn(f'{structure_file}: {str(te)}', Warning)
    return structures


def load_input(user_input, resdir):
    workflow_inp = {}
    workflow = load_workflow(user_input.get('workflow'))
    # check resdir
    # resdir = os.path.abspath(userinput.pop('resdir'))
    if isinstance(resdir, str):
        if hasattr(workflow, 'SUB'):
            for submeta in workflow.SUB:
                workflow_inp.update({submeta: {'resdir':
                                                   os.path.abspath(resdir)}})
        else:
            workflow_inp.update({'resdir': os.path.abspath(resdir)})
    else:
        raise ValueError('`resdir` is not a valid path')
    # check metadata
    if user_input.get('metadata'):
        subdata, metadata = _load_metadata(user_input.pop('metadata'))
        workflow_inp.update(metadata)
        if hasattr(workflow, 'SUB'):
            for submeta in workflow.SUB:
                workflow_inp[submeta].update(subdata)
        else:
            workflow_inp.update(subdata)
    # check subdata
    if user_input.get('subdata'):
        subdata = user_input.pop('subdata')
        for submeta, subinfo in subdata.items():
            if submeta not in workflow.SUB:
                raise KeyError(f'Unknown {submeta} '
                               f'in {user_input.get("workflow")}')
            else:
                workflow_inp[submeta].update(**_load_subdata(subinfo))
    # check structure
    # workflow_inp.update(load_s(userinput))
    return workflow_inp


def load_s(user_input):
    skeys = {'format', 'cell', 'pbc', 'masses'}
    structure_files = user_input.get('structure')
    options = user_input.get('options') or {}
    sargs = {k: v for k, v in user_input.items() if k in skeys}
    # parse structure
    if structure_files:
        workflow_inp = {}
        if isinstance(structure_files, str) and os.path.isfile(structure_files):
            workflow_inp.update({'structure': load_structure(structure_files,
                                                             **sargs,
                                                             **options)})
        elif isinstance(structure_files, list):
            if len(structure_files) < 2:
                raise ValueError('The input `structure` list '
                                 'should be at least two')
            structures = {}
            for i, structure_file in enumerate(structure_files):
                structure = load_structure(structure_file, **sargs, **options)
                structures.update({f'image_{i}': structure})
            workflow_inp.update({'structures': structures})
        else:
            raise ValueError('No valid `structure`, '
                             'it can be a single structure or '
                             'a list of structures')
    else:
        raise KeyError('You need set structure')
    return workflow_inp


# def check_dict(mapping):
#     """Purpose for undoing aiida default serializer,
#     to avoid OrderedDict Error"""
#     if isinstance(mapping, dict):
#         return mapping
#     else:
#         return "Unrecognized dictionary"


class Ecint(WorkChain):
    @classmethod
    def define(cls, spec):
        super(Ecint, cls).define(spec)
        spec.input('webhook', valid_type=(str, type(None)), required=False,
                   non_db=True)
        spec.input('workflow', valid_type=str, required=True, non_db=True)
        spec.input('workflow_inp', valid_type=dict, required=False, non_db=True)

        spec.outline(
            cls.submit_workchain,
            if_(cls.should_notification)(
                cls.inspect_workchain,
                cls.notification
            )
        )

    def should_notification(self):
        return True if self.inputs.get('webhook') else False

    def submit_workchain(self):
        self.report(f'Your workflow is: {self.inputs.workflow}\n'
                    f'Your input is: {self.inputs.workflow_inp}')
        node = self.submit(load_workflow(self.inputs.workflow),
                           **self.inputs.workflow_inp)
        self.to_context(workchain=node)

    def inspect_workchain(self):
        assert self.ctx.workchain.is_terminated

    def notification(self):
        notification_in_dingtalk(self.inputs.webhook, self.ctx.workchain)


def check_webhook(webhook):
    dingtalk_webhook_start = ('https://oapi.dingtalk.com/robot/send'
                              '?access_token=')
    if webhook:
        if isinstance(webhook, str):
            if not webhook.startswith(dingtalk_webhook_start):
                raise ValueError('Not valid dingtalk webhook')
            else:
                return webhook
        else:
            raise ValueError('`webhook` format error')
    else:
        warn('You have not set webhook, so no notification will send', Warning)


def _submit_ecint(resdir, webhook, workflow, workflow_inp):
    """

    Args:
        resdir (str): results directory for workflow
        webhook (str): webhook for notification
        workflow (str): workflow name
        workflow_inp (dict): workflow input,
            e.g. {'structure':, 'resdir':, 'config':,
                  'kind_section':, 'machine':}

    Returns:
        None

    """
    # before submit, check resdir
    if not os.path.exists(resdir):
        os.makedirs(resdir)
    # store all StructureData before submit
    if workflow_inp.get('structure'):
        workflow_inp['structure'].store()
    elif workflow_inp.get('structures'):
        if isinstance(workflow_inp['structures'], dict):
            for structure_data in workflow_inp['structures'].values():
                structure_data.store()
        elif isinstance(workflow_inp['structures'], list):
            for structure_data in workflow_inp['structures']:
                structure_data.store()
    elif workflow_inp.get('imd'):
        for settings in workflow_inp['imd']:
            for structure in settings['structures']:
                structure.store()
    node = submit(Ecint, **{'webhook': webhook,
                            'workflow': workflow,
                            'workflow_inp': workflow_inp})
    with open(os.path.join(resdir, RESULT_NAME), 'a') as f:
        f.write(f'# Your work directory is {os.getcwd()}, PK: {node.pk}\n')


def get_userinput(input_file):
    # load input
    # try:
    #     ecint_input = load_json(input_file)
    # except JSONDecodeError:
    #     ecint_input = load_yaml(input_file)
    ecint_input = load_config(input_file)
    UserInput = create_userinput(ecint_input['workflow'])
    userinput = UserInput(**ecint_input)
    # check webhook
    check_webhook(userinput.webhook)
    return userinput


def submit_from_file(input_file):
    """

    Args:
        input_file (str): path of input file

    Returns:
        None

    """
    userinput = get_userinput(input_file)
    # submit...
    resdir = userinput.resdir
    webhook = userinput.webhook
    workflow = userinput.workflow
    workflow_inp = userinput.get_workflow_inp()
    if isinstance(workflow_inp, dict):
        print('START SUBMIT...')
        # print(workflow_inp)
        _submit_ecint(resdir=resdir,
                      webhook=webhook,
                      workflow=workflow,
                      workflow_inp=workflow_inp)
        print('END SUBMIT')
    elif isinstance(workflow_inp, list):
        print('START SUBMIT MULTI STRUCTURES...')
        for i, one_workflow_inp in enumerate(tqdm(workflow_inp)):
            _submit_ecint(resdir=os.path.join(resdir, str(i)),
                          webhook=webhook,
                          workflow=workflow,
                          workflow_inp=one_workflow_inp)
        print('END SUBMIT MULTI STRUCTURES')
    # return userinput.get_workflow_inp()


@click.command()
@click.argument('filename', type=click.Path(exists=True), default='ecint.json')
def main(filename):
    submit_from_file(filename)
