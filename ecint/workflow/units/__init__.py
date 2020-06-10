import os
from aiida_cp2k.calculations import Cp2kCalculation
from aiida.orm import Dict, StructureData, Code
from aiida.engine import run
from ecint.preprocessor.utils import load_json, load_machine, check_neb

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))


class JsonDryRun(object):
    def __init__(self, inputclass, machine=None):
        self.structure = StructureData(ase=inputclass.structure)
        self.parameters = Dict(dict=inputclass.input_sets)
        self.machine = machine or self.default_machine

    @property
    def default_machine(self):
        machine = {
            'n': 144,
            'W': 20 * 60,
            'q': 'large',
            'ptile': 24,
            'code@computer': 'cp2k@chenglab51',
        }
        return load_machine(machine)

    def load_machine(self, machine_config):
        self.machine = load_json(machine_config)

    @property
    def builder(self):
        builder = Cp2kCalculation.get_builder()
        builder.structure = self.structure
        builder.parameters = self.parameters
        builder.code = Code.get_from_string(self.machine['code@computer'])
        builder.metadata.options.resources = {'tot_num_mpiprocs': self.machine['tot_num_mpiprocs']}
        builder.metadata.options.max_wallclock_seconds = self.machine['max_wallclock_seconds']
        builder.metadata.options.queue_name = self.machine['queue_name']
        builder.metadata.options.custom_scheduler_commands = self.machine['custom_scheduler_commands']

        builder.metadata.dry_run = True
        check_neb(self.parameters.attributes, self.machine)
        return builder

    def run(self):
        run(self.builder)
