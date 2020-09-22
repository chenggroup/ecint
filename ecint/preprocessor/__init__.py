from abc import ABCMeta, abstractmethod

from aiida.orm import Code, Dict, StructureData
from aiida_cp2k.workchains import Cp2kBaseWorkChain
from aiida_deepmd.calculations.dp import DpCalculation
from ase import Atoms

from ecint.preprocessor.utils import get_procs_per_node_from_code_name, \
    load_machine, uniform_neb

__all__ = ['EnergyPreprocessor', 'GeooptPreprocessor', 'NebPreprocessor',
           'FrequencyPreprocessor', 'DeepmdPreprocessor']


class Preprocessor(metaclass=ABCMeta):

    def __init__(self, inpclass, restrict_machine=None):
        """
        
        Args:
            TODO: add ecint.preprocessor.input.BaseInput
            inpclass (Any):
                input class in ecint.preprocessor.input
            restrict_machine (dict):
                restrict machine, assign which code need be used and
                how many calculation resources are required
            
        """
        # self.structure = inpclass.structure
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
    # TODO: make general Preprocessor for another job scheduler,
    #  now just for LSF
    def __init__(self, inpclass, restrict_machine=None):
        super(Cp2kPreprocessor, self).__init__(inpclass, restrict_machine)
        self.structure = inpclass.structure

    @property
    def builder(self):
        _builder = Cp2kBaseWorkChain.get_builder()
        if isinstance(self.structure, StructureData):
            _builder.cp2k.structure = self.structure
        elif isinstance(self.structure, Atoms):
            _builder.cp2k.structure = StructureData(ase=self.structure)
        _builder.cp2k.parameters = \
            self.parameters
        _builder.cp2k.code = \
            Code.get_from_string(self.machine.get('code@computer'))
        _builder.cp2k.metadata.options.resources = \
            {'tot_num_mpiprocs': self.machine.get('tot_num_mpiprocs')}
        if self.machine.get('max_wallclock_seconds'):
            _builder.cp2k.metadata.options.max_wallclock_seconds = \
                self.machine.get('max_wallclock_seconds')
        if self.machine.get('queue_name'):
            _builder.cp2k.metadata.options.queue_name = \
                self.machine.get('queue_name')
        if self.machine.get('custom_scheduler_commands'):
            _builder.cp2k.metadata.options.custom_scheduler_commands = \
                self.machine.get('custom_scheduler_commands')
        return _builder


class DeepmdPreprocessor(Preprocessor):
    def __init__(self, inpclass, restrict_machine=None):
        super(DeepmdPreprocessor, self).__init__(inpclass, restrict_machine)
        self.datadirs = inpclass.datadirs

    @property
    def builder(self):
        _builder = DpCalculation.get_builder()
        if isinstance(self.datadirs, list):
            _builder.datadirs = self.datadirs
        # place the input parameters
        _builder.loss = Dict(dict=self.parameters['loss'])
        _builder.training = Dict(dict=self.parameters['training'])
        _builder.learning_rate = Dict(dict=self.parameters['learning_rate'])
        _builder.model = Dict(dict=self.parameters['model'])

        # machine information
        _builder.code = \
            Code.get_from_string(self.machine.get('code@computer'))
        _builder.metadata.options.resources = {
            'num_machines': 1,
            'tot_num_mpiprocs': self.machine.get('tot_num_mpiprocs')
        }
        if self.machine.get('max_wallclock_seconds'):
            _builder.metadata.options.max_wallclock_seconds = \
                self.machine.get('max_wallclock_seconds')
        if self.machine.get('queue_name'):
            _builder.metadata.options.queue_name = \
                self.machine.get('queue_name')
        if self.machine.get('custom_scheduler_commands'):
            _builder.metadata.options.custom_scheduler_commands = \
                self.machine.get('custom_scheduler_commands')
        return _builder


class EnergyPreprocessor(Cp2kPreprocessor):
    @property
    def builder(self):
        builder = super(EnergyPreprocessor, self).builder
        builder.cp2k.settings = Dict(
            dict={'additional_retrieve_list': ["*.cube", "*.pdos"]})
        return builder


class GeooptPreprocessor(Cp2kPreprocessor):
    @property
    def builder(self):
        builder = super(GeooptPreprocessor, self).builder
        builder.cp2k.settings = Dict(
            dict={'additional_retrieve_list': ["*-pos-1.xyz"]})
        return builder


class NebPreprocessor(Cp2kPreprocessor):
    @property
    def builder(self):
        builder = super(NebPreprocessor, self).builder
        builder.cp2k.settings = Dict(
            dict={'additional_retrieve_list': ["*-pos-Replica_nr_?-1.xyz"]})
        # uniform_neb(self.parameters.attributes, self.machine)
        return builder


class FrequencyPreprocessor(Cp2kPreprocessor):
    @property
    def builder(self):
        builder = super(FrequencyPreprocessor, self).builder
        # builder.settings = \
        #     Dict(dict={'additional_retrieve_list': ["aiida.out"]})
        # aiida.out is already in retrieve_list
        return builder
