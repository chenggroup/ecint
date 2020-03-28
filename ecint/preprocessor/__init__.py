from abc import ABCMeta, abstractmethod
from ecint.preprocessor.utils import *
from aiida.plugins import CalculationFactory
from aiida.orm import Dict, Code, StructureData
from ecint.preprocessor.utils import load_json


class Preprocessor(metaclass=ABCMeta):
    """
    input: BaseInput(structure) object
    machine: bsub setting, dict or json file
    """

    def __init__(self, inputclass, machine=None):
        self.structure = StructureData(ase=inputclass.structure)
        self.parameters = Dict(dict=inputclass.input_sets)
        self.machine = machine

    def get_machine_from_json(self, machine_file_path):
        self.machine = load_json(machine_file_path)

    @property
    def builder(self):
        pass


class LSFPreprocessor(Preprocessor):
    def builder(self):
        code = self.machine['code@computer']
        tot_num_mpiprocs = self.machine['n']
        max_wallclock_seconds = self.machine['W']
        queue_name = self.machine['q']
        custom_scheduler_commands = self.machine['R']

        Cp2kCalculation = CalculationFactory('cp2k')
        builder = Cp2kCalculation.get_builder()
        builder.structure = self.structure
        builder.parameters = self.parameters
        builder.code = Code.get_from_string(code)
        builder.metadata.options.resources = {'tot_num_mpiprocs': tot_num_mpiprocs}
        builder.metadata.options.max_wallclock_seconds = max_wallclock_seconds
        builder.metadata.options.queue_name = queue_name
        builder.metadata.options.custom_scheduler_commands = f'#BSUB -R \"{custom_scheduler_commands}\"'
        return builder
