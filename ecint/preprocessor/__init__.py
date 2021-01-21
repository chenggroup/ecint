from abc import ABCMeta, abstractmethod

from aiida.orm import Code, Dict, StructureData
from aiida_cp2k.workchains import Cp2kBaseWorkChain
from aiida_deepmd.calculations.dp import DpCalculation
from aiida_lammps.calculations.lammps.template import BatchTemplateCalculation
from ase import Atoms

from ecint.preprocessor.utils import get_procs_per_node_from_code_name, \
    load_machine, uniform_neb

__all__ = ['EnergyPreprocessor', 'GeooptPreprocessor', 'NebPreprocessor',
           'FrequencyPreprocessor', 'DPPreprocessor', 'QBCPreprocessor']


def set_machine(builder, restrict_machine, isslurm=False):
    """

    Args:
        builder (aiida.engine.processes.builder.ProcessBuilder):
        restrict_machine (dict):
        isslurm (bool):

    Returns:
        aiida.engine.processes.builder.ProcessBuilder

    """
    builder.code = \
        Code.get_from_string(restrict_machine.get('code@computer'))
    builder.metadata.options.resources = {
        'tot_num_mpiprocs': restrict_machine.get('tot_num_mpiprocs')
    }
    # TODO (@robin): whether let users to change num_machines or not
    # TODO: make general Preprocessor for another job scheduler,
    #  now just for LSF, SLURM
    if isslurm:
        builder.metadata.options.resources.update({'num_machines': 1})
    if restrict_machine.get('max_wallclock_seconds'):
        builder.metadata.options.max_wallclock_seconds = \
            restrict_machine.get('max_wallclock_seconds')
    if restrict_machine.get('queue_name'):
        builder.metadata.options.queue_name = \
            restrict_machine.get('queue_name')
    if restrict_machine.get('custom_scheduler_commands'):
        builder.metadata.options.custom_scheduler_commands = \
            restrict_machine.get('custom_scheduler_commands')
    return builder


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
        _builder.cp2k.parameters = self.parameters

        set_machine(_builder['cp2k'], self.machine)
        return _builder


class DPPreprocessor(Preprocessor):
    def __init__(self, inpclass, restrict_machine=None):
        super(DPPreprocessor, self).__init__(inpclass, restrict_machine)
        self.datadirs = inpclass.datadirs
        self.kinds = inpclass.kinds
        self.descriptor_sel = inpclass.descriptor_sel

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
        set_machine(_builder, self.machine, isslurm=True)
        return _builder


class QBCPreprocessor(Preprocessor):
    def __init__(self, inpclass, restrict_machine=None):
        super(QBCPreprocessor, self).__init__(inpclass, restrict_machine)
        self.structures = inpclass.structures
        self.kinds = inpclass.kinds

    @property
    def builder(self):
        _builder = BatchTemplateCalculation.get_builder()
        _builder.structures = self.structures
        _builder.kinds = self.kinds
        _builder.template = self.parameters['template']
        _builder.variables = self.parameters['variables']
        _builder.file = self.parameters['file']
        _builder.settings = Dict(
            dict={'additional_retrieve_list': ['*/model_devi.out']})

        set_machine(_builder, self.machine, isslurm=True)
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
