import os
from dataclasses import dataclass, field, asdict
from warnings import warn

import click
from aiida import load_profile
from aiida.engine import WorkChain, submit, if_
from ase.io.formats import UnknownFileTypeError
from tqdm import tqdm

from ecint.config import RESULT_NAME
from ecint.postprocessor.utils import notification_in_dingtalk
from ecint.preprocessor.kind import KindSection
from ecint.preprocessor.utils import load_json, load_yaml, \
    load_config, load_structure, load_kind, load_machine

load_profile()


@dataclass
class SubData(object):
    config: str or dict = None
    kind_section: str or dict or list = None
    machine: str or dict = None


@dataclass
class UserInput(object):
    workflow: str
    webhook: str = None
    resdir: str = field(default=os.getcwd())
    # structure section
    structures_folder: str = None  # if set, high-throughput calc will run, conflicting with `structure`
    structure: str or list = None
    cell: list = field(default_factory=list)
    pbc: bool or list = True
    metadata: dict = field(default_factory=dict)  # SubData and other special paras
    subdata: dict = field(default_factory=dict)  # {str: SubData(config, kind_section, machine), ...}

    @property
    def has_structures_folder(self):
        if self.structure and self.structures_folder:
            raise KeyError('`structure` and `structures_folder` can not coexist')
        return True if self.structures_folder else False

    def get_workflow_inp(self):
        if self.has_structures_folder:
            workflow_inp = []
            if os.path.isdir(self.structures_folder):
                for i, structure_file in enumerate(tqdm(os.listdir(self.structures_folder))):
                    try:
                        structure_dir = os.path.join(self.structures_folder, structure_file)
                        workflow_inp.append({'structure': load_structure(structure_dir, self.cell, self.pbc),
                                             **load_input(asdict(self), resdir=os.path.join(self.resdir, str(i)))})
                    except UnknownFileTypeError as te:
                        warn(f'{structure_file}: {str(te)}', Warning)
            else:
                raise ValueError('`structures_folder` is not a folder')
        else:
            workflow_inp = {**load_s(asdict(self)), **load_input(asdict(self), resdir=self.resdir)}
        return workflow_inp


def _load_subdata(user_input):
    workflow_inp = {}
    # check config
    if user_input.get('config'):
        config = user_input.pop('config')
        workflow_inp.update({'config': load_config(config)})
    # check kind_section
    if user_input.get('kind_section'):
        kind_section = user_input.pop('kind_section')
        if isinstance(kind_section, dict) and ('BASIS_SET' in kind_section) and ('POTENTIAL' in kind_section):
            _kind_section = KindSection(kind_section['BASIS_SET'], kind_section['POTENTIAL'])
            workflow_inp.update({'kind_section': _kind_section})
        elif isinstance(kind_section, dict) and set(kind_section) != {'BASIS_SET', 'POTENTIAL'}:
            raise ValueError('If you setup `kind_setion`, you need set both `BASIS_SET` and `POTENTIAL`')
        else:
            workflow_inp.update({'kind_section': load_kind(kind_section)})
    # check machine
    if user_input.get('machine'):
        machine = user_input.pop('machine')
        workflow_inp.update({'machine': load_machine(machine)})
    return workflow_inp


def _load_metadata(user_input):
    return {**_load_subdata(user_input)}, {**user_input}


def load_input(user_input, resdir):
    workflow_inp = {}
    workflow = eval(user_input.get('workflow'))
    # check resdir
    # resdir = os.path.abspath(userinput.pop('resdir'))
    if os.path.isdir(resdir):
        if hasattr(workflow, 'SUB'):
            for submeta in workflow.SUB:
                workflow_inp.update({submeta: {'resdir': os.path.abspath(resdir)}})
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
                raise KeyError(f'Unknown {submeta} in {user_input.get("workflow")}')
            else:
                workflow_inp[submeta].update(_load_subdata(subinfo))
    # check structure
    # workflow_inp.update(load_s(userinput))
    return workflow_inp


def load_s(user_input):
    structure_files = user_input.get('structure')
    cell = user_input.get('cell')
    pbc = user_input.get('pbc')
    # parse structure
    if structure_files:
        workflow_inp = {}
        if isinstance(structure_files, str) and os.path.isfile(structure_files):
            workflow_inp.update({'structure': load_structure(structure_files, cell, pbc)})
        elif isinstance(structure_files, list):
            if len(structure_files) < 2:
                raise ValueError('The input `structure` list should be at least two')
            structures = {}
            for i, structure_file in enumerate(structure_files):
                structure = load_structure(structure_file, cell, pbc)
                structures.update({f'image_{i}': structure})
            workflow_inp.update({'structures': structures})
        else:
            raise ValueError('No valid `structure`, it can be a single structure or a list of structures')
    else:
        raise KeyError('You need set structure')
    return workflow_inp


class Ecint(WorkChain):
    @classmethod
    def define(cls, spec):
        super(Ecint, cls).define(spec)
        spec.input('webhook', valid_type=(str, type(None)), required=False, non_db=True)
        spec.input('workflow', valid_type=str, required=True, non_db=True)
        spec.input('workflow_inp', valid_type=dict, required=True, non_db=True)

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
        node = self.submit(eval(self.inputs.workflow), **self.inputs.workflow_inp)
        self.to_context(workchain=node)

    def inspect_workchain(self):
        assert self.ctx.workchain.is_terminated

    def notification(self):
        notification_in_dingtalk(self.inputs.webhook, self.ctx.workchain)


def check_webhook(webhook):
    dingtalk_webhook_start = 'https://oapi.dingtalk.com/robot/send?access_token='
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
        resdir ():
        webhook ():
        workflow ():
        workflow_inp ():

    Returns:

    """
    # before submit, check resdir
    if not os.path.exists(resdir):
        os.makedirs(resdir)
    # store all StructureData before submit
    if workflow_inp.get('structure'):
        workflow_inp['structure'].store()
    else:  # userinput.get_workflow_inp().get('structures')
        for structure_data in workflow_inp['structures'].values():
            structure_data.store()
    node = submit(Ecint, **{'webhook': webhook,
                            'workflow': workflow,
                            'workflow_inp': workflow_inp})
    with open(os.path.join(resdir, RESULT_NAME), 'a') as f:
        f.write(f'# Your work directory is {os.getcwd()}, PK: {node.pk}\n')


def submit_from_file(input_file):
    """

    Args:
        input_file (str): path of input file

    Returns:
        None

    """
    # load input
    ecint_input = load_json(input_file) if input_file.endswith('.json') else load_yaml(input_file)
    userinput = UserInput(**ecint_input)
    # check webhook
    webhook = userinput.webhook
    check_webhook(webhook)
    # check resdir and make it if not exists
    if not os.path.exists(userinput.resdir):
        print(f'{userinput.resdir} not exist, so auto make it')
        os.mkdir(userinput.resdir)
    # submit...
    workflow = userinput.workflow
    workflow_inp = userinput.get_workflow_inp()
    if isinstance(workflow_inp, dict):
        print('START SUBMIT...')
        _submit_ecint(resdir=userinput.resdir,
                      webhook=webhook,
                      workflow=workflow,
                      workflow_inp=workflow_inp)
        print('END SUBMIT')
    elif isinstance(workflow_inp, list):
        print('START SUBMIT MULTI STRUCTURES')
        for i, one_workflow_inp in enumerate(tqdm(workflow_inp)):
            _submit_ecint(resdir=os.path.join(userinput.resdir, str(i)),
                          webhook=webhook,
                          workflow=workflow,
                          workflow_inp=one_workflow_inp)
        print('END SUBMIT MULTI STRUCTURES')
    # return userinput.get_workflow_inp()


@click.command()
@click.argument('input_file', type=click.Path(exists=True), default='neb.json')
def main(input_file):
    submit_from_file(input_file)
