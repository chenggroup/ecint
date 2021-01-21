from aiida import load_profile
from aiida.engine import submit, run
from aiida.orm import StructureData
from aiida_cp2k.calculations import Cp2kCalculation
from ase import Atoms
from ase.io import read
from ecint.preprocessor.utils import load_machine

from ecint.config import default_cp2k_machine, default_cp2k_large_machine, default_dpmd_gpu_machine, default_lmp_gpu_machine
from ecint.preprocessor import (Preprocessor, set_machine,
                                LammpsPreprocessor, DeepmdPreprocessor)
from ecint.preprocessor.input import *

load_profile()


def convert_structures_in_dict(sdict: dict):
    for k, s in sdict.items():
        atoms = read(s)
        structure = StructureData(ase=atoms)
        sdict[k] = structure
    return sdict


class Cp2kPreprocessor(Preprocessor):
    def __init__(self, inpclass, restrict_machine=None):
        super(Cp2kPreprocessor, self).__init__(inpclass, restrict_machine)
        self.structure = inpclass.structure

    @property
    def builder(self):
        _builder = Cp2kCalculation.get_builder()
        if isinstance(self.structure, StructureData):
            _builder.structure = self.structure
        elif isinstance(self.structure, Atoms):
            _builder.structure = StructureData(ase=self.structure)
        _builder.parameters = self.parameters

        set_machine(_builder, self.machine)
        return _builder


class TestPreprocessor(object):
    def __init__(self, preprocessor, inpclass, restrict_machine=None):
        self.preprocessor = preprocessor(inpclass, restrict_machine)

    @property
    def builder(self):
        _builder = self.preprocessor.builder
        _builder.metadata.dry_run = True
        return _builder

    def dry_run(self):
        _node = submit(self.builder)
        return _node.dry_run_info['folder']


if __name__ == '__main__':
    inp = NebInputSets(**convert_structures_in_dict(
        {'structure': 'resources/ethane_1.xyz'}))
    pre = TestPreprocessor(preprocessor=Cp2kPreprocessor,
                           inpclass=inp,
                           restrict_machine=load_machine(default_cp2k_large_machine))
    output_folder = pre.dry_run()
    print(output_folder)
