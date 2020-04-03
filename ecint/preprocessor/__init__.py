from abc import ABCMeta, abstractmethod
from ecint.preprocessor.utils import *
from aiida.plugins import WorkflowFactory
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

    def load_machine_from_json(self, machine_file_path):
        self.machine = load_json(machine_file_path)

    @property
    @abstractmethod
    def builder(self):
        pass


class LSFPreprocessor(Preprocessor):
    @property
    def builder(self):
        code = self.machine['code@computer']
        tot_num_mpiprocs = self.machine['n']
        max_wallclock_seconds = self.machine['W']
        queue_name = self.machine['q']
        custom_scheduler_commands = self.machine['R']

        Cp2kBaseWorkChain = WorkflowFactory('cp2k.base')
        builder = Cp2kBaseWorkChain.get_builder()
        builder.cp2k.structure = self.structure
        builder.cp2k.parameters = self.parameters
        builder.cp2k.code = Code.get_from_string(code)
        builder.cp2k.metadata.options.resources = {'tot_num_mpiprocs': tot_num_mpiprocs}
        builder.cp2k.metadata.options.max_wallclock_seconds = max_wallclock_seconds
        builder.cp2k.metadata.options.queue_name = queue_name
        builder.cp2k.metadata.options.custom_scheduler_commands = f'#BSUB -R \"{custom_scheduler_commands}\"'
        return builder


class EnergyPreprocessor(LSFPreprocessor):
    @property
    def builder(self):
        builder = super(EnergyPreprocessor, self).builder
        return builder


class GeooptPreprocessor(LSFPreprocessor):
    @property
    def builder(self):
        builder = super(GeooptPreprocessor, self).builder
        builder.cp2k.settings = Dict(dict={'additional_retrieve_list': ["*-pos-1.xyz"]})
        return builder


class NebPreprocessor(LSFPreprocessor):
    @property
    def builder(self):
        builder = super(NebPreprocessor, self).builder
        builder.cp2k.settings = Dict(dict={'additional_retrieve_list': ["*-pos-Replica_nr_?-1.xyz"]})
        return builder


class FrequencyPreprocessor(LSFPreprocessor):
    @property
    def builder(self):
        builder = super(FrequencyPreprocessor, self).builder
        # builder.settings = Dict(dict={'additional_retrieve_list': ["aiida.out"]})
        # aiida.out is already in retrieve_list
        return builder
