from time import sleep
from ase import Atoms
from ase.io import write
from ecint.postprocessor import get_last_frame
from aiida.common import AttributeDict
from aiida.engine import submit, ToContext
from aiida.orm import Dict
from aiida_cp2k.calculations import Cp2kCalculation
from aiida_cp2k.workchains import Cp2kBaseWorkChain
from aiida.engine import while_, if_
from aiida.plugins import CalculationFactory
# from ecint.preprocessor import GeooptPreprocessor, NebPreprocessor, FrequencyPreprocessor
from ecint.preprocessor.input import InputSetsFromFile
from ecint.preprocessor import LSFPreprocessor
from ecint.preprocessor.input import GeooptInputSets, NebInputSets, FrequencyInputSets
from aiida_cp2k.workchains.aiida_base_restart import BaseRestartWorkChain
from ase.io import read
from aiida.plugins import WorkflowFactory
from aiida.orm import (Code, Dict, SinglefileData)
from aiida.engine import run
from aiida import load_profile
import numpy as np

load_profile()
# Cp2kCalculation = CalculationFactory('cp2k')


class NebInputSets(InputSetsFromFile):
    def __init__(self, structure, config='neb.json', kind_section_config='DZVPSets'):
        super(NebInputSets, self).__init__(structure, config, kind_section_config)


class NebPreprocessor(LSFPreprocessor):
    @property
    def builder(self):
        builder = super(NebPreprocessor, self).builder
        builder.settings = Dict(dict={'additional_retrieve_list': ["*-pos-Replica_nr_?-1.xyz"]})
        react_file = SinglefileData(file='R.xyz')
        product_file = SinglefileData(file='P.xyz')
        builder.file = {'react': react_file, 'product': product_file}
        # builder.metadata.dry_run = True  # 不提交至服务器，仅模拟提交生成输入文件，正式工作流中需删除
        return builder


class NebWorkChain(Cp2kBaseWorkChain):
    @classmethod
    def define(cls, spec):
        super(NebWorkChain, cls).define(spec)
        spec.expose_inputs(Cp2kBaseWorkChain, namespace='cp2k')
        spec.input('atoms', valid_type=Atoms, required=False)
        spec.input('machine', valid_type=dict, required=False)

        # spec.input('structure_list')  # 反应物，生成物，(其他点)

        spec.outline(
            cls.submit_workchain,
            if_(not cls.ctx.workchain.is_finished_ok)(
                while_(not cls.ctx.workchain.is_finished_ok)(sleep(5*60)),
                # 下面两个方法是关于后处理的部分
                cls.get_traj_for_energy_curve,
                cls.get_max_energy_frame,
            ),
        )

        spec.expose_outputs(Cp2kBaseWorkChain)

    def submit_workchain(self):
        inputclass = NebInputSets(self.inputs.atoms)
        builder = NebPreprocessor(inputclass, self.inputs.machine).builder
        future = self.submit(builder)
        return ToContext(workchain=future)

    def get_traj_for_energy_curve(self):
        # self.outputs.rechieved
        traj_list = [f'aiida-1-pos-Replica_{n}r_1-1.xyz' for n in range(1, 7)]
        traj_list_for_energy_curve = []
        for replica_traj in traj_list:
            last_frame = get_last_frame(replica_traj)
            traj_list_for_energy_curve.append(last_frame)
        write('aiida-Replica_5r.xyz', traj_list_for_energy_curve)
        return traj_list_for_energy_curve

    def get_max_energy_frame(self):
        energy_list = np.array([traj.info['E'] for traj in self.ctx.traj_list])
        e_max = energy_list.argmax()
        return e_max


if __name__ == '__main__':
    pass