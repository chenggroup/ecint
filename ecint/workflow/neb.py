from os import chdir
from os.path import join, abspath, isabs
from warnings import warn
from numpy import savetxt
from ase.io import read, write
from aiida.orm import StructureData
from aiida.engine import WorkChain, ExitCode, ToContext, if_
from ecint.preprocessor import test_machine
from ecint.preprocessor.utils import load_json, load_machine, check_neb, inspect_node, is_valid_workdir
from ecint.workflow.units import CONFIG_DIR
from ecint.workflow.units.base import GeooptSingleWorkChain, NebSingleWorkChain, FrequencySingleWorkChain


class NebWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super(NebWorkChain, cls).define(spec)
        # use structures.image_0 as reactant, image_1 as next point in energy curve, and so on
        # the last image_N as product
        spec.input_namespace('structures', valid_type=StructureData, dynamic=True)
        spec.input('workdir', valid_type=(str, type(None)), default=None, required=False, non_db=True,
                   validator=is_valid_workdir)

        # set global input files, default is None
        spec.input('kind_section_file', valid_type=(str, type(None)), default=None, required=False, non_db=True)
        spec.input('machine_file', valid_type=(str, type(None)), default=None, required=False, non_db=True)

        # set geoopt input files
        spec.expose_inputs(GeooptSingleWorkChain, namespace='geoopt', exclude=['structure'])
        # spec.input('geoopt.config_file', default=join(CONFIG_DIR, 'geoopt.json'),
        #            valid_type=str, required=False, non_db=True)
        # spec.input('geoopt.kind_section_file', valid_type=str, default='DZVPBLYP', required=False, non_db=True)
        # spec.input('geoopt.machine_file', valid_type=str, default='AutoMode', required=False, non_db=True)

        # set neb input files
        spec.expose_inputs(NebSingleWorkChain, namespace='neb', exclude=['structures'])
        # spec.input('neb.config_file', default=join(CONFIG_DIR, 'neb.json'),
        #            valid_type=str, required=False, non_db=True)
        # spec.input('neb.kind_section_file', valid_type=str, default='DZVPBLYP', required=False, non_db=True)
        # spec.input('neb.machine_file', valid_type=str, default='AutoMode', required=False, non_db=True)

        # set frequency input files
        spec.expose_inputs(FrequencySingleWorkChain, namespace='frequency', exclude=['structure'])
        # spec.input('frequency.config_file', default=join(CONFIG_DIR, 'frequency.json'),
        #            valid_type=str, required=False, non_db=True)
        # spec.input('frequency.kind_section_file', valid_type=str, default='DZVPBLYP', required=False, non_db=True)
        # spec.input('frequency.machine_file', valid_type=str, default='AutoMode', required=False, non_db=True)

        spec.outline(
            cls.check_machine_and_config,
            cls.submit_geoopt,
            cls.inspect_geoopt,
            cls.submit_neb,
            cls.inspect_neb,
            cls.submit_frequency,
            cls.inspect_frequency,
            if_(cls.validate_workdir)(
                cls.write_outputs,
            ),
        )

        spec.expose_outputs(GeooptSingleWorkChain, namespace='reactant')
        spec.expose_outputs(GeooptSingleWorkChain, namespace='product')
        spec.expose_outputs(NebSingleWorkChain)
        spec.expose_outputs(FrequencySingleWorkChain)

    def check_machine_and_config(self):
        if len(self.inputs.structures) < 2:
            raise ValueError('The input structures should be at least two--reactant and product')
        # use self.ctx.neb_config to replace
        self.ctx.neb_config = load_json(self.inputs.neb.config_file)
        self.ctx.number_of_replica = self.ctx.neb_config['MOTION']['BAND']['NUMBER_OF_REPLICA']
        if self.ctx.number_of_replica < len(self.inputs.structures):
            raise ValueError('Number of input structures should be greater than number of replicas'
                             'which you set in /MOTION/BAND/NUMBER_OF_REPLICA')
        if self.inputs.neb.machine_file == 'AutoMode':
            auto_machine = {'code@computer': 'cp2k@aiida_test', 'nnode': self.ctx.number_of_replica, 'queue': 'large'}
            self.ctx.neb_machine = load_machine(auto_machine)
        elif self.inputs.neb.machine_file == 'TestMode':
            self.ctx.neb_machine = load_machine(test_machine)
        else:
            self.ctx.neb_machine = load_machine(self.inputs.neb.machine_file)
        self.ctx.machine = None
        if self.inputs.machine_file:
            self.ctx.machine = load_machine(self.inputs.machine_file)
            self.report(f'Use {abspath(self.inputs.machine_file)} as machine config')
        check_neb(self.ctx.neb_config, self.ctx.machine or self.ctx.neb_machine)

    def submit_geoopt(self):
        reactant = self.inputs.structures['image_0']
        self.ctx.image_last_index = len(self.inputs.structures) - 1
        product = self.inputs.structures[f'image_{self.ctx.image_last_index}']
        node_reactant = self.submit(GeooptSingleWorkChain, structure=reactant,
                                    **self.exposed_inputs(GeooptSingleWorkChain, namespace='geoopt'))
        node_product = self.submit(GeooptSingleWorkChain, structure=product,
                                   **self.exposed_inputs(GeooptSingleWorkChain, namespace='geoopt'))
        return ToContext(geoopt_reactant_workchain=node_reactant, geoopt_product_workchain=node_product)

    def inspect_geoopt(self):
        inspect_node(self.ctx.geoopt_reactant_workchain)
        self.out_many(
            self.exposed_outputs(self.ctx.geoopt_reactant_workchain, GeooptSingleWorkChain, namespace='reactant')
        )
        inspect_node(self.ctx.geoopt_product_workchain)
        self.out_many(
            self.exposed_outputs(self.ctx.geoopt_product_workchain, GeooptSingleWorkChain, namespace='product')
        )

    def submit_neb(self):
        structures_geoopt = {}
        reactant_geoopt = self.ctx.geoopt_reactant_workchain.outputs.structure_geoopt
        product_geoopt = self.ctx.geoopt_product_workchain.outputs.structure_geoopt
        structures_geoopt.update({'image_0': reactant_geoopt, f'image_{self.ctx.image_last_index}': product_geoopt})
        for image_index in range(1, self.ctx.image_last_index):
            structures_geoopt.update({f'image_{image_index}': self.inputs.structures[f'image_{image_index}']})
        node = self.submit(NebSingleWorkChain, structures=structures_geoopt,
                           **self.exposed_inputs(NebSingleWorkChain, namespace='neb'))
        self.to_context(neb_workchain=node)

    def inspect_neb(self):
        inspect_node(self.ctx.neb_workchain)
        self.out_many(
            self.exposed_outputs(self.ctx.neb_workchain, NebSingleWorkChain)
        )

    def submit_frequency(self):
        transition_state = self.ctx.neb_workchain.outputs.transition_state
        node = self.submit(FrequencySingleWorkChain, structure=transition_state,
                           **self.exposed_inputs(FrequencySingleWorkChain, namespace='frequency'))
        self.to_context(frequency_workchain=node)

    def inspect_frequency(self):
        inspect_node(self.ctx.frequency_workchain)
        self.out_many(
            self.exposed_outputs(self.ctx.frequency_workchain, FrequencySingleWorkChain)
        )

    def validate_workdir(self):
        return self.inputs.workdir

    def write_outputs(self):
        chdir(self.inputs.workdir)
        # write geoopt structures
        reactant_geoopt = self.outputs['reactant']['structure_geoopt']
        reactant_geoopt_atoms = reactant_geoopt.get_ase()
        reactant_geoopt_atoms.info.update({'E': reactant_geoopt.get_attribute('energy')})
        product_geoopt = self.outputs['product']['structure_geoopt']
        product_geoopt_atoms = product_geoopt.get_ase()
        product_geoopt_atoms.info.update({'E': product_geoopt.get_attribute('energy')})
        reactant_geoopt_atoms.write('reactant_geoopt.xyz')
        product_geoopt_atoms.write('product_geoopt.xyz')
        # write trajactory for energy curve
        traj_data = self.outputs['traj_for_energy_curve']
        energy_array = traj_data.get_array('energy')
        traj = []
        for structure_index in traj_data.get_stepids():
            structure = traj_data.get_step_structure(structure_index)
            energy = energy_array[structure_index]
            atoms = structure.get_ase()
            atoms.info.update({'i': structure_index, 'E': energy})
            traj.append(atoms)
        write('traj_for_energy_curve.xyz', traj)
        # write transition state structure
        transition_state = self.outputs['transition_state']
        transition_state.get_ase().write('transition_state.xyz')
        # write vibrational frequency value
        freq_data = self.outputs['vibrational_frequency']
        freq_list = freq_data.get_list()
        savetxt('frequency.txt', freq_list, fmt='%-15s%-15s%-15s', header='VIB|Frequency (cm^-1)')
