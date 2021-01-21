import os
import os
import random
from collections import namedtuple
from uuid import uuid4

from aiida.engine import while_, WorkChain
from aiida.orm import Int, SinglefileData, StructureData
from ase.io import read

from ecint.config import RESULT_NAME
from ecint.postprocessor.utils import write_datadir_from_energyworkchain
from ecint.postprocessor.visualization import get_learning_curve
from ecint.preprocessor.utils import inspect_node
from ecint.workflow.units.base import DPSingleWorkChain, \
    EnergySingleWorkChain, QBCBatchWorkChain


def download_file(node, remotename, localname):
    """
    use node.called[0] to get calculation node

    Args:
        node:
        remotename:
        localname:

    Returns:

    """
    if os.path.dirname(localname):
        os.makedirs(os.path.dirname(localname), exist_ok=True)
    node_cal = node.called[0]
    node_cal.outputs.remote_folder.getfile(remotename,
                                           os.path.abspath(localname))


class ActiveLearning(WorkChain):
    TYPE = 'mixing'
    SUB = {'training', 'exploration', 'labeling'}
    _ITER_NAME = 'iter_'
    _MODELS_DIR = 'models'
    _MD_DIR = 'md'
    _SCREEN_DIR = 'screen'
    _FP_DIR = 'fp'

    @classmethod
    def define(cls, spec):
        super(ActiveLearning, cls).define(spec)
        # spec.inputs('structures', valid_type=StructureData)
        spec.expose_inputs(DPSingleWorkChain,
                           include=['datadirs', 'kinds', 'descriptor_sel'])
        # `structures` and many `variables` in imd
        # TODO: different structures with different variables in a same loop
        spec.input('imd', valid_type=list, non_db=True)
        spec.input('num_pb', default=Int(4), required=False, valid_type=Int)
        # TODO: need parameters like fp_task_max/min
        spec.input('labeling.task_max', valid_type=int, default=20, non_db=True)
        spec.input('labeling.task_min', valid_type=int, default=1, non_db=True)
        spec.expose_inputs(DPSingleWorkChain,
                           namespace='training',
                           include=['resdir', 'config', 'machine'])
        spec.expose_inputs(QBCBatchWorkChain,
                           namespace='exploration',
                           include=['resdir', 'template', 'machine',
                                    'model_devi'])
        spec.expose_inputs(EnergySingleWorkChain,
                           namespace='labeling',
                           include=['resdir', 'config', 'machine',
                                    'kind_section'])

        spec.outline(
            cls.check_imd,
            cls.init_settings,
            while_(cls.should_active_learning)(
                cls.submit_training,
                cls.inspect_training,
                cls.write_models,
                cls.submit_exploration,
                cls.inspect_exploration,
                cls.get_candidate,
                cls.submit_labeling,
                cls.inspect_labeling,
                cls.get_datadir,
            ),
            cls.submit_training,
            cls.inspect_training,
            cls.write_models,
            cls.get_pbs,
            cls.write_results
        )

        spec.output_namespace('models', valid_type=SinglefileData, dynamic=True)

    def check_imd(self):
        for settings in self.inputs.imd:
            for v in settings.values():
                if not isinstance(v, list):
                    raise TypeError('Variables in imd must be list')

    def init_settings(self):
        # init loops count
        self.ctx.loops = -1
        # datadirs is type list
        self.ctx.datadirs = self.inputs.datadirs

    def should_active_learning(self):
        if self.ctx.loops == len(self.inputs.imd) - 1:
            self.ctx.loops += 1
            return False
        elif self.ctx.loops < len(self.inputs.imd) - 1:
            self.ctx.loops += 1
            return True
        else:
            self.report(f'loops: {self.ctx.loops} is bigger than '
                        f'len(imd): {len(self.inputs.imd)}')
            return False

    def submit_training(self):
        for i in range(self.inputs.num_pb.value):
            init_inputs = self.exposed_inputs(DPSingleWorkChain)
            init_inputs['datadirs'] = self.ctx.datadirs
            node = self.submit(DPSingleWorkChain,
                               **init_inputs,
                               **self.exposed_inputs(DPSingleWorkChain,
                                                     namespace='training',
                                                     agglomerate=False))
            self.to_context(**{f'dpmd_{i}': node})

    def inspect_training(self):
        for i in range(self.inputs.num_pb.value):
            inspect_node(self.ctx[f'dpmd_{i}'])

    def write_models(self):
        self.report(f'Loops {self.ctx.loops}')
        models_dir = os.path.join(self.inputs.training.resdir,
                                  self._ITER_NAME + str(self.ctx.loops),
                                  self._MODELS_DIR)
        # model.pb and lcurve.out
        for i in range(self.inputs.num_pb.value):
            model_number_name = os.path.join(models_dir, str(i))
            os.makedirs(model_number_name, exist_ok=True)
            model_name = os.path.join(model_number_name, 'model.pb')
            lcurve_name = os.path.join(model_number_name, 'lcurve.out')
            # write model.pb
            with self.ctx[f'dpmd_{i}'].outputs.model.open(mode='rb') as f:
                model_content = f.read()
            with open(model_name, mode='wb') as f:
                f.write(model_content)
            # write lcurve.out
            lcurve_content = self.ctx[f'dpmd_{i}'].outputs.lcurve.get_content()
            with open(lcurve_name, mode='w') as f:
                f.write(lcurve_content)
        # datadirs
        datadirs_name = os.path.join(models_dir, 'datadirs')
        os.makedirs(datadirs_name, exist_ok=True)
        for datadir in self.ctx.datadirs:
            dest_datadir = os.path.join(datadirs_name,
                                        os.path.basename(datadir))
            if not os.path.exists(dest_datadir):
                os.symlink(datadir, dest_datadir)
        # force_learning_curve.jpg
        lcurve_list = [os.path.join(models_dir, str(i), 'lcurve.out')
                       for i in range(self.inputs.num_pb.value)]
        get_learning_curve(lcurve_list,
                           os.path.join(models_dir, 'force_lcurve.jpg'))

    def get_pbs(self):
        for i in range(self.inputs.num_pb.value):
            self.out(f'models.dpmd_{i}', self.ctx[f'dpmd_{i}'].outputs.model)

    def submit_exploration(self):
        # nloop = 0  # should be changed when run workchain, now just test
        md_dir = os.path.join(self.inputs.exploration.resdir,
                              self._ITER_NAME + str(self.ctx.loops),
                              self._MD_DIR)
        os.makedirs(md_dir, exist_ok=True)
        variables = self.inputs.imd[self.ctx.loops]
        structures = variables.pop('structures')
        node = self.submit(QBCBatchWorkChain,
                           label=md_dir,
                           structures=structures,
                           variables=variables,
                           kinds=self.inputs.kinds,
                           graphs=[self.ctx[f'dpmd_{i}'].outputs.model
                                   for i in range(self.inputs.num_pb.value)],
                           **self.exposed_inputs(QBCBatchWorkChain,
                                                 namespace='exploration',
                                                 agglomerate=False))
        self.to_context(qbc=node)
        # # construct s_and_v for a dict like {'s': [], 'v': [], ...}
        # s_and_v = {}
        # TODO: move it (isinstance) to check_imd
        # for k in settings.keys():
        #     if isinstance(settings[k], list):
        #         s_and_v.update({k: settings[k]})
        #     else:
        #         raise TypeError(f'`{k}` should be list')
        # # just split v, and keep s
        # self.ctx.s_list = s_and_v.pop('structures')
        # self.ctx.v_list = [dict(zip(s_and_v.keys(), v))
        #                    for v in product(*s_and_v.values())]
        # # submit and get n * model_devi.out
        # self.ctx.md_dir = os.path.join(self.inputs.exploration.resdir,
        #                                self._ITER_NAME + str(self.ctx.loops),
        #                                self._MD_DIR)
        # for i_v, v in enumerate(self.ctx.v_list):
        #     for i_s, s in enumerate(self.ctx.s_list):
        #         sv = {'structure': s, 'variables': v}
        #         condition_dir = os.path.join(self.ctx.md_dir,
        #                                      f'condition_{i_v}')
        #         os.makedirs(condition_dir, exist_ok=True)
        #         node = self.submit(QBCBatchWorkChain,
        #                            label=os.path.join(condition_dir,
        #                                               f'model_devi_{i_s}'),
        #                            kinds=self.inputs.kinds,
        #                            graphs=[self.ctx[f'dpmd_{i}'].outputs.model
        #                                    for i in
        #                                    range(self.inputs.num_pb.value)],
        #                            **sv,
        #                            **self.exposed_inputs(
        #                                QBCBatchWorkChain,
        #                                namespace='exploration'))
        #         self.to_context(**{f'model_devi_{i_v}_{i_s}': node})

    def inspect_exploration(self):
        inspect_node(self.ctx.qbc)

    # TODO: one ssh connect to download all lmp structures
    def get_candidate(self):
        # select candidate
        p = namedtuple('Point', ['i_traj', 'step'])
        all_candidate, selected_candidate = [], None
        for i_traj, traj in enumerate([md_index['candidate'] for md_index in
                                       self.ctx.qbc.outputs.model_devi_index]):
            for step in traj:
                all_candidate.append(p(i_traj, step))
        if len(all_candidate) > self.inputs.labeling.task_max:
            selected_candidate = random.sample(all_candidate,
                                               k=self.inputs.labeling.task_max)
        elif ((len(all_candidate) >= self.inputs.labeling.task_min) and
              (len(all_candidate) <= self.inputs.labeling.task_max)):
            selected_candidate = all_candidate
        elif len(all_candidate) < self.inputs.labeling.task_min:
            # TODO: break big loop
            self.report('No enough candidates')  # set flag to break whole loops
        # download *.lammpstrj
        self.ctx.fp_dir = os.path.join(self.inputs.labeling.resdir,
                                       self._ITER_NAME + str(self.ctx.loops),
                                       self._FP_DIR)
        self.ctx.fp_stc = []
        if selected_candidate:
            for candidate in selected_candidate:
                remote_folder = self.ctx.qbc.outputs.remote_folder
                subfolder = str(candidate.i_traj)
                filename = str(candidate.step) + '.lammpstrj'
                localfolder = os.path.join(self.ctx.fp_dir, 'candidate',
                                           subfolder)
                localfile = os.path.join(localfolder, filename)
                os.makedirs(localfolder, exist_ok=True)
                remote_folder.getfile(os.path.join(subfolder, filename),
                                      localfile)
                atoms = read(localfile, format='lammps-dump-text',
                             specorder=self.inputs.kinds)
                self.ctx.fp_stc.append(StructureData(ase=atoms))

        # for i_v, condition in enumerate(self.ctx.v_list):
        #     condition_dir = os.path.join(self.ctx.md_dir, f'condition_{i_v}')
        #     # condition.json
        #     with open(os.path.join(condition_dir, 'condition.json'), 'w') as f:
        #         json.dump(condition, f, sort_keys=True, indent=2)
        #     # distribution.jpg
        #     model_devi_list = [os.path.join(condition_dir,
        #                                     f'model_devi_{i_s}.out')
        #                        for i_s in range(len(self.ctx.s_list))]
        #     fl = self.inputs.exploration.model_devi.force_low_limit
        #     fh = self.inputs.exploration.model_devi.force_high_limit
        #     skip = self.inputs.exploration.model_devi.skip_images
        #     get_model_devi_distribution(
        #         model_devi_list, fl, fh, skip,
        #         'Distribution of force deviation',
        #         os.path.join(condition_dir, 'force_devi_distribution.jpg'))
        #
        # # select candidate
        # p = namedtuple('Point', ['i_v', 'i_s', 'step'])
        # all_candidate, selected_candidate = [], None
        # for i_v in range(len(self.ctx.v_list)):
        #     for i_s in range(len(self.ctx.s_list)):
        #         node = self.ctx[f'model_devi_{i_v}_{i_s}']
        #         for step in node.outputs.candidate:
        #             all_candidate.append(p(i_v, i_s, step))
        # if len(all_candidate) > self.inputs.labeling.task_max:
        #     selected_candidate = random.sample(all_candidate,
        #                                        k=self.inputs.labeling.task_max)
        # elif ((len(all_candidate) >= self.inputs.labeling.task_min) and
        #       (len(all_candidate) <= self.inputs.labeling.task_max)):
        #     selected_candidate = all_candidate
        # elif len(all_candidate) < self.inputs.labeling.task_min:
        #     # TODO: break big loop
        #     self.report('No enough candidates')  # set flag to break whole loops
        # # download *.lammpstrj
        # self.ctx.structures = []
        # if selected_candidate:
        #     for candidate in selected_candidate:
        #         i_v, i_s, step = candidate
        #         node = self.ctx[f'model_devi_{i_v}_{i_s}']
        #         filename = f'{step}.lammpstrj'
        #         file_download = os.path.join(self.ctx.md_dir,
        #                                      f'condition_{i_v}',
        #                                      f'traj_{i_s}',
        #                                      filename)
        #         download_file(node, filename, file_download)
        #         atoms = read(file_download, format='lammps-dump-text',
        #                      specorder=self.inputs.kinds)
        #         self.ctx.structures.append(StructureData(ase=atoms))

    def submit_labeling(self):
        for i, structure in enumerate(self.ctx.fp_stc):
            node = self.submit(EnergySingleWorkChain,
                               label=os.path.join(self.ctx.fp_dir,
                                                  f'coords_{i}'),
                               structure=structure,
                               **self.exposed_inputs(EnergySingleWorkChain,
                                                     namespace='labeling'))
            self.to_context(**{f'fp_{i}': node})

    def inspect_labeling(self):
        for i in range(len(self.ctx.fp_stc)):
            inspect_node(self.ctx[f'fp_{i}'])

    def get_datadir(self):
        next_models_dir = os.path.join(self.inputs.training.resdir,
                                       (self._ITER_NAME +
                                        str(self.ctx.loops + 1)),
                                       self._MODELS_DIR)
        datadirs_name = os.path.join(next_models_dir, 'datadirs')
        os.makedirs(datadirs_name, exist_ok=True)
        nodes = [self.ctx[f'fp_{i}'] for i in range(len(self.ctx.fp_stc))]
        nodes_categories = {}
        for node in nodes:
            structure = node.inputs.structure
            tr_mark = ''.join([str(self.inputs.kinds.index(kindname))
                               for kindname in structure.get_site_kindnames()])
            if tr_mark in nodes_categories:
                nodes_categories[tr_mark].append(node)
            else:
                nodes_categories[tr_mark] = [node]
        for nodes_category in nodes_categories.values():
            data_dirname = os.path.join(datadirs_name,
                                        f'loop{self.ctx.loops}_{uuid4().hex}')
            write_datadir_from_energyworkchain(dirname=data_dirname,
                                               nodes=nodes_category,
                                               kinds=self.inputs.kinds)
            self.ctx.datadirs.append(data_dirname)

    def write_results(self):
        os.chdir(self.inputs.training.resdir)
        loop_dir = os.path.join(self.inputs.training.resdir,
                                self._ITER_NAME + str(self.ctx.loops))
        with open(RESULT_NAME, 'a') as f:
            f.write('# END ACTIVE LEARNING\n')
            f.write(f'Last loop: {loop_dir}\n')
