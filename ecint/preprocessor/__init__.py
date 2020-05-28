from abc import ABCMeta, abstractmethod

from aiida.orm import Dict, Code, StructureData
# from aiida.plugins import WorkflowFactory
from aiida_cp2k.workchains import Cp2kBaseWorkChain

from ecint.preprocessor.utils import load_machine, get_procs_per_node_from_code_name, check_neb


class Preprocessor(metaclass=ABCMeta):
    """
    input: BaseInput(structure) object
    machine: bsub setting, dict or json file
    """

    def __init__(self, inputclass, machine=None):
        self.structure = StructureData(ase=inputclass.structure)
        self.parameters = Dict(dict=inputclass.input_sets)
        self.machine = machine

    def load_machine_from_json(self, json_path):
        self.machine = load_machine(json_path)

    @property
    @abstractmethod
    def builder(self):
        pass


class LSFPreprocessor(Preprocessor):
    @property
    def builder(self):
        builder = Cp2kBaseWorkChain.get_builder()
        builder.cp2k.structure = self.structure
        builder.cp2k.parameters = self.parameters
        builder.cp2k.code = Code.get_from_string(self.machine['code@computer'])
        builder.cp2k.metadata.options.resources = {'tot_num_mpiprocs': self.machine['tot_num_mpiprocs']}
        builder.cp2k.metadata.options.max_wallclock_seconds = self.machine['max_wallclock_seconds']
        builder.cp2k.metadata.options.queue_name = self.machine['queue_name']
        builder.cp2k.metadata.options.custom_scheduler_commands = self.machine['custom_scheduler_commands']
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
        check_neb(self.parameters.attributes, self.machine)
        return builder


class FrequencyPreprocessor(LSFPreprocessor):
    @property
    def builder(self):
        builder = super(FrequencyPreprocessor, self).builder
        # builder.settings = Dict(dict={'additional_retrieve_list': ["aiida.out"]})
        # aiida.out is already in retrieve_list
        return builder
