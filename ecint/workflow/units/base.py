import re
from os.path import join, abspath

import numpy as np
from aiida.engine import WorkChain
from aiida.orm import StructureData, TrajectoryData, List
from ase.io import read

from ecint.preprocessor import GeooptPreprocessor, NebPreprocessor, \
    FrequencyPreprocessor
from ecint.preprocessor.input import GeooptInputSets, NebInputSets, FrequencyInputSets
from ecint.preprocessor.utils import load_json, load_machine, inspect_node, check_neb
from ecint.workflow.units import CONFIG_DIR


class BaseSingleWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super(BaseSingleWorkChain, cls).define(spec)
        spec.input('structure', valid_type=StructureData, required=False)
        # need add config_file in sub singleworkchain, e.g.
        # spec.input('config_file', default=join(CONFIG_DIR, 'config.json'),
        #            valid_type=str, required=False, non_db=True)
        spec.input('kind_section_file', valid_type=str, default='DZVPBLYP', required=False, non_db=True)
        spec.input('machine_file', valid_type=str, default='AutoMode', required=False, non_db=True)

        """
        define spec.outline like follows in sub singleworkchain
        spec.outline(
            cls.check_config,
            cls.check_machine,
            cls.submit_worchain,
            cls.inspect_worchain,
            cls.get_results,
        )
        """

        # need add spec.output in sub singleworkchain

    def check_machine(self):
        if self.inputs.machine_file == 'AutoMode':
            self.ctx.machine = load_machine(default_machine)
            self.report('Auto setup machine config')
        elif self.inputs.machine_file == 'TestMode':
            self.ctx.machine = load_machine(test_machine)
            self.report('Use test config, please make sure you are testing')
        else:
            self.ctx.machine = load_machine(self.inputs.machine_file)
            self.report(f'Use {abspath(self.inputs.machine_file)} as machine config')


class GeooptSingleWorkChain(BaseSingleWorkChain):
    @classmethod
    def define(cls, spec):
        super(GeooptSingleWorkChain, cls).define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('config_file', default=join(CONFIG_DIR, 'geoopt.json'),
                   valid_type=str, required=False, non_db=True)

        spec.outline(
            cls.check_machine,
            cls.submit_geoopt,
            cls.inspect_geoopt,
            cls.get_structure_geoopt,
        )

        spec.output('structure_geoopt')  # structure after geoopt

    def submit_geoopt(self):
        inputclass = GeooptInputSets(structure=self.inputs.structure,
                                     config=self.inputs.config_file,
                                     kind_section_config=self.inputs.kind_section_file)
        pre = GeooptPreprocessor(inputclass, self.ctx.machine)
        builder = pre.builder
        node = self.submit(builder)
        self.to_context(geoopt_workchain=node)

    def inspect_geoopt(self):
        inspect_node(self.ctx.geoopt_workchain)

    def get_structure_geoopt(self):
        structure_geoopt = self.ctx.geoopt_workchain.outputs.output_structure.clone()
        energy = self.ctx.geoopt_workchain.outputs.output_parameters.get_attribute('energy')
        structure_geoopt.set_attribute('energy', energy)
        self.out('structure_geoopt', structure_geoopt.store())


class NebSingleWorkChain(BaseSingleWorkChain):
    @classmethod
    def define(cls, spec):
        super(NebSingleWorkChain, cls).define(spec)
        # use structures.image_0 as reactant, image_1 as next point in energy curve, and so on
        # the last image_N as product
        spec.input_namespace('structures', valid_type=StructureData, dynamic=True)
        spec.input('config_file', default=join(CONFIG_DIR, 'neb.json'),
                   valid_type=str, required=False, non_db=True)

        spec.outline(
            cls.check_config,
            cls.check_machine,
            cls.set_cell_and_pbc,
            cls.submit_neb,
            cls.inspect_neb,
            cls.get_energy_curve_data,
            cls.get_transition_state,
        )

        spec.output('traj_for_energy_curve')
        spec.output('transition_state')

    def check_config(self):
        if len(self.inputs.structures) < 2:
            raise ValueError('The input structures should be at least two--reactant and product')
        self.ctx.neb_config = load_json(self.inputs.config_file)
        self.ctx.number_of_replica = self.ctx.neb_config['MOTION']['BAND']['NUMBER_OF_REPLICA']
        if self.ctx.number_of_replica < len(self.inputs.structures):
            raise ValueError('Number of input structures should be greater than number of replicas'
                             'which you set in /MOTION/BAND/NUMBER_OF_REPLICA')

    def check_machine(self):
        super(NebSingleWorkChain, self).check_machine()
        if self.inputs.machine_file == 'AutoMode':
            auto_machine = {'code@computer': 'cp2k@aiida_test', 'nnode': self.ctx.number_of_replica, 'queue': 'large'}
            self.ctx.machine = load_machine(auto_machine)
        check_neb(self.ctx.neb_config, self.ctx.machine)

    def set_cell_and_pbc(self):
        self.ctx.reactant = self.inputs.structures['image_0']
        self.ctx.cell = self.ctx.reactant.cell
        self.ctx.pbc = self.ctx.reactant.pbc

    def submit_neb(self):
        # pre setup inputclass
        inputclass = NebInputSets(structure=self.ctx.reactant,
                                  config=self.inputs.config_file,
                                  kind_section_config=self.inputs.kind_section_file)
        # add input structures, then submit them to remote
        for image_index in range(len(self.inputs.structures)):
            inputclass.add_config({'MOTION': {'BAND': {'REPLICA': [{'COORD_FILE_NAME': f'image_{image_index}.xyz'}]}}})

        pre = NebPreprocessor(inputclass, self.ctx.machine)
        builder = pre.builder
        builder.cp2k.file = self.inputs.structures
        node = self.submit(builder)
        self.to_context(neb_workchain=node)

    def inspect_neb(self):
        inspect_node(self.ctx.neb_workchain)

    def get_energy_curve_data(self):
        node = self.ctx.neb_workchain
        # get list of `Atoms`
        replica_traj = []
        energy_list = []
        for i in range(1, self.ctx.number_of_replica + 1):
            # warning: if project name is not 'aiida', this part will fail,
            # 'aiida' is the _DEFAULT_PROJECT_NAME in aiida_cp2k.calculations.Cp2kCalculation
            with node.outputs.retrieved.open(f'aiida-pos-Replica_nr_{i}-1.xyz') as replica_file:
                replica_atoms = read(replica_file, index='-1', format='xyz')
                replica_atoms.set_cell(self.ctx.cell)
                replica_atoms.set_pbc(self.ctx.pbc)
                energy = replica_atoms.info['E']
                replica = StructureData(ase=replica_atoms)
                energy_list.append(energy)
                replica_traj.append(replica)
        self.ctx.traj_for_energy_curve = TrajectoryData(structurelist=replica_traj)
        self.ctx.traj_for_energy_curve.set_array(name='energy', array=np.array(energy_list))
        self.out('traj_for_energy_curve', self.ctx.traj_for_energy_curve.store())  # output type TrajectoryData

    def get_transition_state(self):
        traj_data = self.ctx.traj_for_energy_curve
        structure_list = [traj_data.get_step_structure(i) for i in traj_data.get_stepids()]
        energy_array = traj_data.get_array('energy')
        structure_with_max_energy = structure_list[energy_array.argmax()]
        max_energy = energy_array.max()
        structure_with_max_energy.set_attribute('energy', max_energy)
        self.out('transition_state', structure_with_max_energy.store())  # output type StructureData


class FrequencySingleWorkChain(BaseSingleWorkChain):
    @classmethod
    def define(cls, spec):
        super(FrequencySingleWorkChain, cls).define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('config_file', default=join(CONFIG_DIR, 'frequency.json'),
                   valid_type=str, required=False, non_db=True)

        spec.outline(
            cls.check_machine,
            cls.submit_frequency,
            cls.inspect_frequency,
            cls.get_vib_frequency,
        )

        spec.output('vibrational_frequency')  # unit: cm^-1

    def submit_frequency(self):
        inputclass = FrequencyInputSets(structure=self.inputs.structure,
                                        config=self.inputs.config_file,
                                        kind_section_config=self.inputs.kind_section_file)
        pre = FrequencyPreprocessor(inputclass, self.ctx.machine)
        builder = pre.builder
        node = self.submit(builder)
        self.to_context(frequency_workchain=node)

    def inspect_frequency(self):
        inspect_node(self.ctx.frequency_workchain)

    def get_vib_frequency(self):
        node = self.ctx.frequency_workchain
        output_content = node.outputs.retrieved.get_object_content('aiida.out')
        frequency_str_data = re.findall(r'VIB\|Frequency.*', output_content)
        frequency_float_data = [list(map(float, data.strip('VIB|Frequency (cm^-1)').split()))
                                for data in frequency_str_data]
        frequency_data = List(list=frequency_float_data)
        self.out('vibrational_frequency', frequency_data.store())  # output type List
