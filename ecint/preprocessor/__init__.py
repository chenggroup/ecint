from abc import ABCMeta, abstractmethod

from aiida.orm import Dict, Code, StructureData
from aiida_cp2k.workchains import Cp2kBaseWorkChain
from ase import Atoms

from ecint.preprocessor.utils import load_machine, get_procs_per_node_from_code_name, uniform_neb


class Preprocessor(metaclass=ABCMeta):

    def __init__(self, inpclass, restrict_machine=None):
        """
        
        Args:
            inpclass (ecint.preprocessor.input.BaseInput): input class in ecint.preprocessor.input
            restrict_machine (dict): restrict machine,
                            assign which code need be used and how many calculation resources are required
            
        """
        self.structure = inpclass.structure
        self.parameters = Dict(dict=inpclass.input_sets)
        self.machine = restrict_machine

    def load_machine(self, machine):
        """load general machine to restrict machine
        """
        self.machine = load_machine(machine)

    @property
    @abstractmethod
    def builder(self):
        """Set up aiida.engine.WorkChain.get_builder()
        """
        pass


class Cp2kPreprocessor(Preprocessor):
    # TODO: make general Preprocessor for another job scheduler, now just for LSF
    @property
    def builder(self):
        builder = Cp2kBaseWorkChain.get_builder()
        if isinstance(self.structure, StructureData):
            builder.cp2k.structure = self.structure
        elif isinstance(self.structure, Atoms):
            builder.cp2k.structure = StructureData(ase=self.structure)
        builder.cp2k.parameters = self.parameters
        builder.cp2k.code = Code.get_from_string(self.machine.get('code@computer'))
        builder.cp2k.metadata.options.resources = {'tot_num_mpiprocs': self.machine.get('tot_num_mpiprocs')}
        if self.machine.get('max_wallclock_seconds'):
            builder.cp2k.metadata.options.max_wallclock_seconds = self.machine.get('max_wallclock_seconds')
        if self.machine.get('queue_name'):
            builder.cp2k.metadata.options.queue_name = self.machine.get('queue_name')
        if self.machine.get('custom_scheduler_commands'):
            builder.cp2k.metadata.options.custom_scheduler_commands = self.machine.get('custom_scheduler_commands')
        return builder


class EnergyPreprocessor(Cp2kPreprocessor):
    @property
    def builder(self):
        builder = super(EnergyPreprocessor, self).builder
        return builder


class GeooptPreprocessor(Cp2kPreprocessor):
    @property
    def builder(self):
        builder = super(GeooptPreprocessor, self).builder
        builder.cp2k.settings = Dict(dict={'additional_retrieve_list': ["*-pos-1.xyz"]})
        return builder


class NebPreprocessor(Cp2kPreprocessor):
    @property
    def builder(self):
        builder = super(NebPreprocessor, self).builder
        builder.cp2k.settings = Dict(dict={'additional_retrieve_list': ["*-pos-Replica_nr_?-1.xyz"]})
        # uniform_neb(self.parameters.attributes, self.machine)
        return builder


class FrequencyPreprocessor(Cp2kPreprocessor):
    @property
    def builder(self):
        builder = super(FrequencyPreprocessor, self).builder
        # builder.settings = Dict(dict={'additional_retrieve_list': ["aiida.out"]})
        # aiida.out is already in retrieve_list
        return builder
