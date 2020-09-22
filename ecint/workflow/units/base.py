import os
import re

import numpy as np
from aiida.engine import WorkChain
from aiida.orm import Float, List, SinglefileData, StructureData, TrajectoryData

from ecint.config import default_cp2k_large_machine, default_cp2k_machine, \
    default_dpmd_gpu_machine, RESULT_NAME
from ecint.postprocessor.utils import AU2EV, get_last_frame, \
    write_xyz_from_structure, write_xyz_from_trajectory
from ecint.postprocessor.visualization import plot_energy_curve
from ecint.preprocessor import *
from ecint.preprocessor.input import *
from ecint.preprocessor.kind import DZVPPBE, KindSection
from ecint.preprocessor.utils import check_config_machine, inspect_node, \
    uniform_neb

__all__ = ['EnergySingleWorkChain', 'GeooptSingleWorkChain',
           'NebSingleWorkChain', 'FrequencySingleWorkChain',
           'DeepmdSingleWorkChain']


# def load_default_config(config_name):
#     return load_config(os.path.join(CONFIG_DIR, config_name))


class BaseSingleWorkChain(WorkChain):
    TYPE = 'simulation'

    @classmethod
    def define(cls, spec):
        super(BaseSingleWorkChain, cls).define(spec)
        spec.input('resdir', valid_type=str, required=True, non_db=True)
        # to distinguish different file name,
        # `label` used inside WorkChain, it is unnecessary for user input
        spec.input('label', default='',
                   valid_type=str, required=False, non_db=True)
        # structure or structures
        # spec.input('structure', valid_type=StructureData, required=False)
        # need add config in sub singleworkchain, e.g.
        spec.input('config', default='default',
                   valid_type=(str, dict), required=False, non_db=True)
        spec.input('kind_section', default=DZVPPBE(),
                   valid_type=(list, KindSection), required=False, non_db=True)
        spec.input('machine', default=default_cp2k_machine,
                   valid_type=dict, required=False, non_db=True)

        # define spec.outline like follows in sub singleworkchain
        """
        spec.outline(
            cls.check_config_machine,
            cls.submit_workchain,
            cls.inspect_workchain,
            cls.get_results,
            cls.write_results,
        )
        """

        # need add spec.output in sub singleworkchain

    def check_config_machine(self):
        self.ctx.config, self.ctx.machine = \
            check_config_machine(self.inputs.config, self.inputs.machine)

    # def submit_workchain(self):
    #     inp = UnitsInputSets(structure=self.inputs.structure,
    #                          config=self.ctx.config,
    #                          kind_section=self.inputs.kind_section)
    #     pre = Cp2kPreprocessor(inp, self.ctx.machine)
    #     builder = pre.builder
    #     node = self.submit(builder)
    #     self.to_context(workchain=node)
    #
    # def inspect_workchain(self):
    #     inspect_node(self.ctx.workchain)


class EnergySingleWorkChain(BaseSingleWorkChain):
    @classmethod
    def define(cls, spec):
        super(EnergySingleWorkChain, cls).define(spec)
        spec.input('label', default='coords',
                   valid_type=str, required=False, non_db=True)
        spec.input('structure',
                   valid_type=StructureData, required=True)

        spec.outline(
            cls.check_config_machine,
            cls.submit_energy,
            cls.inspect_energy,
            cls.get_energy,
            cls.write_results
        )

        spec.output('energy', valid_type=Float)

    def submit_energy(self):
        inp = EnergyInputSets(structure=self.inputs.structure,
                              config=self.ctx.config,
                              kind_section=self.inputs.kind_section)
        pre = EnergyPreprocessor(inp, self.ctx.machine)
        builder = pre.builder
        node = self.submit(builder)
        self.to_context(energy_workchain=node)

    def inspect_energy(self):
        inspect_node(self.ctx.energy_workchain)

    def get_energy(self):
        self.ctx.energy = (self.ctx.energy_workchain.outputs.
                           output_parameters.get_attribute('energy') * AU2EV)
        energy_data = Float(self.ctx.energy)
        self.out('energy', energy_data.store())

    def write_results(self):
        os.chdir(self.inputs.resdir)
        # write structure with energy
        output_structure_name = f'{self.inputs.label}.xyz'
        atoms = self.inputs.structure.get_ase()
        atoms.info.update({'E': f'{self.ctx.energy} eV'})
        atoms.write(output_structure_name)

        with open(RESULT_NAME, 'a') as f:
            f.write(f'# Step: Energy, PK: {self.ctx.energy_workchain.pk}\n')
            f.write(f'energy (eV): {self.ctx.energy}\n')


class GeooptSingleWorkChain(BaseSingleWorkChain):
    @classmethod
    def define(cls, spec):
        super(GeooptSingleWorkChain, cls).define(spec)
        spec.input('label', default='structure',
                   valid_type=str, required=False, non_db=True)
        spec.input('structure',
                   valid_type=StructureData, required=True)

        spec.outline(
            cls.check_config_machine,
            cls.submit_geoopt,
            cls.inspect_geoopt,
            cls.get_structure_geoopt,
            cls.write_results
        )

        spec.output('structure_geoopt', valid_type=StructureData)

    def submit_geoopt(self):
        inp = GeooptInputSets(structure=self.inputs.structure,
                              config=self.ctx.config,
                              kind_section=self.inputs.kind_section)
        pre = GeooptPreprocessor(inp, self.ctx.machine)
        builder = pre.builder
        node = self.submit(builder)
        self.to_context(geoopt_workchain=node)

    def inspect_geoopt(self):
        inspect_node(self.ctx.geoopt_workchain)

    def get_structure_geoopt(self):
        retrieved = self.ctx.geoopt_workchain.outputs.retrieved
        traj_regex = re.compile(r'.*-pos-1.xyz')
        traj_file = next(filter(lambda x: traj_regex.match(x),
                                retrieved.list_object_names()))
        with retrieved.open(traj_file) as f:
            geoopt_atoms = get_last_frame(f, cell=self.inputs.structure.cell,
                                          pbc=self.inputs.structure.pbc)
        self.ctx.structure_geoopt = StructureData(ase=geoopt_atoms)
        energy = (self.ctx.geoopt_workchain.outputs.
                  output_parameters.get_attribute('energy') * AU2EV)
        self.ctx.structure_geoopt.set_attribute('energy', energy)
        self.out('structure_geoopt', self.ctx.structure_geoopt.store())

    def write_results(self):
        os.chdir(self.inputs.resdir)
        # write structure file after geoopt
        output_structure_name = f'{self.inputs.label}_geoopt.xyz'
        write_xyz_from_structure(self.ctx.structure_geoopt,
                                 output_file=output_structure_name)

        with open(RESULT_NAME, 'a') as f:
            f.write(f'# Step: Geoopt, PK: {self.ctx.geoopt_workchain.pk}\n')
            f.write(f'structure file: {output_structure_name}\n')
            f.write(f'energy (eV): '
                    f'{self.ctx.structure_geoopt.get_attribute("energy")} eV\n')


class NebSingleWorkChain(BaseSingleWorkChain):
    @classmethod
    def define(cls, spec):
        super(NebSingleWorkChain, cls).define(spec)
        # use structures.image_0 as reactant,
        # image_1 as next point in energy curve, and so on
        # the last image_N as product
        # For provenance graph, need StructureData instead of List
        spec.input_namespace('structures',
                             valid_type=StructureData, dynamic=True)
        spec.input('machine', default=default_cp2k_large_machine,
                   valid_type=dict, required=False, non_db=True)

        spec.outline(
            cls.check_config_machine,
            cls.set_cell_and_pbc,
            cls.submit_neb,
            cls.inspect_neb,
            cls.get_energy_curve_data,
            cls.get_transition_state,
            cls.write_results
        )

        spec.output('traj_for_energy_curve', valid_type=TrajectoryData)
        spec.output('transition_state', valid_type=StructureData)

    def check_config_machine(self):
        self.ctx.config, self.ctx.machine = \
            check_config_machine(self.inputs.config, self.inputs.machine,
                                 uniform_func=uniform_neb)

    def set_cell_and_pbc(self):
        self.ctx.reactant = self.inputs.structures['image_0']
        self.ctx.cell = self.ctx.reactant.cell
        self.ctx.pbc = self.ctx.reactant.pbc

    def submit_neb(self):
        # pre setup inputsets
        inp = NebInputSets(structure=self.ctx.reactant,
                           config=self.ctx.config,
                           kind_section=self.inputs.kind_section)
        # add input structures, then submit them to remote
        for image_index in range(len(self.inputs.structures)):
            inp.add_config({
                'MOTION': {
                    'BAND': {
                        'REPLICA': [{'COORD_FILE_NAME':
                                         f'image_{image_index}.xyz'}]
                    }
                }
            })

        pre = NebPreprocessor(inp, self.ctx.machine)
        builder = pre.builder
        builder.cp2k.file = self.inputs.structures
        node = self.submit(builder)
        self.to_context(neb_workchain=node)

    def inspect_neb(self):
        inspect_node(self.ctx.neb_workchain)

    def get_energy_curve_data(self):
        retrieved = self.ctx.neb_workchain.outputs.retrieved
        # get list of `Atoms`
        replica_regex = re.compile(r'.*-pos-Replica_nr_(\d+)-1.xyz')
        replica_file_list = sorted(filter(lambda x: replica_regex.match(x),
                                          retrieved.list_object_names()),
                                   key=lambda x: int(replica_regex.
                                                     match(x).group(1)))
        replica_traj = []
        energy_list = []
        for replica_file in replica_file_list:
            with retrieved.open(replica_file) as f:
                replica_atoms = get_last_frame(f, cell=self.ctx.cell,
                                               pbc=self.ctx.pbc)
                energy = replica_atoms.info['E'] * AU2EV
                replica = StructureData(ase=replica_atoms)
                energy_list.append(energy)
                replica_traj.append(replica)
        self.ctx.traj_for_energy_curve = \
            TrajectoryData(structurelist=replica_traj)
        self.ctx.traj_for_energy_curve.set_array(name='energy',
                                                 array=np.array(energy_list))
        self.out('traj_for_energy_curve',
                 self.ctx.traj_for_energy_curve.store())

    def get_transition_state(self):
        traj_data = self.ctx.traj_for_energy_curve
        structure_list = [traj_data.get_step_structure(i) for i in
                          traj_data.get_stepids()]
        energy_array = traj_data.get_array('energy')
        self.ctx.structure_with_max_energy = \
            structure_list[energy_array.argmax()]
        max_energy = energy_array.max()
        self.ctx.structure_with_max_energy.set_attribute('energy', max_energy)
        self.out('transition_state',
                 self.ctx.structure_with_max_energy.store())

    def write_results(self):
        os.chdir(self.inputs.resdir)
        # write trajactory for energy curve
        output_traj_name = 'images_traj.xyz'
        write_xyz_from_trajectory(self.ctx.traj_for_energy_curve,
                                  output_file=output_traj_name)
        # plot potential energy curve with data in traj_for_energy_curve
        output_energy_curve_name = 'potential_energy_path.png'
        plot_energy_curve(self.ctx.traj_for_energy_curve,
                          output_file=output_energy_curve_name)
        # write transition state structure
        output_ts_name = 'transition_state.xyz'
        write_xyz_from_structure(self.ctx.structure_with_max_energy,
                                 output_file=output_ts_name)

        with open(RESULT_NAME, 'a') as f:
            f.write(f'# Step: NEB, PK: {self.ctx.neb_workchain.pk}\n')
            f.write(f'trajectory file: {output_traj_name}\n')
            f.write(f'potential energy curve: {output_energy_curve_name}\n')
            f.write(f'transition state file: {output_ts_name}\n')


class FrequencySingleWorkChain(BaseSingleWorkChain):
    @classmethod
    def define(cls, spec):
        super(FrequencySingleWorkChain, cls).define(spec)
        spec.input('structure',
                   valid_type=StructureData, required=True)
        spec.input('machine', default=default_cp2k_large_machine,
                   valid_type=dict, required=False, non_db=True)

        spec.outline(
            cls.check_config_machine,
            cls.submit_frequency,
            cls.inspect_frequency,
            cls.get_vib_frequency,
            cls.write_results
        )

        spec.output('vibrational_frequency')  # type is List, unit is cm^-1

    def submit_frequency(self):
        inp = FrequencyInputSets(structure=self.inputs.structure,
                                 config=self.ctx.config,
                                 kind_section=self.inputs.kind_section)
        pre = FrequencyPreprocessor(inp, self.ctx.machine)
        builder = pre.builder
        node = self.submit(builder)
        self.to_context(frequency_workchain=node)

    def inspect_frequency(self):
        inspect_node(self.ctx.frequency_workchain)

    def get_vib_frequency(self):
        node = self.ctx.frequency_workchain
        output_content = node.outputs.retrieved.get_object_content('aiida.out')
        frequency_str_data = re.findall(r'VIB\|Frequency.*', output_content)
        frequency_float_data = \
            [list(map(float, data.strip('VIB|Frequency (cm^-1)').split()))
             for data in frequency_str_data]
        self.ctx.frequency_data = List(list=frequency_float_data)
        self.out('vibrational_frequency', self.ctx.frequency_data.store())

    def write_results(self):
        os.chdir(self.inputs.resdir)
        # write frequency value
        output_frequency_name = 'frequency.txt'
        freq_list = self.ctx.frequency_data.get_list()
        np.savetxt('frequency.txt', freq_list, fmt='%-15s%-15s%-15s',
                   header='VIB|Frequency (cm^-1)')

        with open(RESULT_NAME, 'a') as f:
            f.write(f'# Step: Frequency, '
                    f'PK: {self.ctx.frequency_workchain.pk}\n')
            f.write(f'frequency file: {output_frequency_name}')


class DeepmdSingleWorkChain(BaseSingleWorkChain):
    TYPE = 'deepmd'

    @classmethod
    def define(cls, spec):
        super(DeepmdSingleWorkChain, cls).define(spec)
        spec.input('datadirs', valid_type=list, required=True, non_db=True)
        spec.input(
            'machine',
            default=default_dpmd_gpu_machine,
            valid_type=dict,
            required=False,
            non_db=True
        )

        spec.outline(
            cls.check_config_machine,
            cls.submit_dpmd,
            cls.inspect_dpmd,
            cls.get_pb,
            cls.write_results
        )

        spec.output('model.pb', valid_type=SinglefileData)

    def submit_dpmd(self):
        inp = DeepmdInputSets(
            datadirs=self.inputs.datadirs,
            config=self.ctx.config
        )
        pre = DeepmdPreprocessor(inp, self.ctx.machine)
        builder = pre.builder
        node = self.submit(builder)
        self.to_context(dpmd_workchain=node)

    def inspect_dpmd(self):
        inspect_node(self.ctx.dpmd_workchain)

    def get_pb(self):
        model = (self.ctx.dpmd_workchain.outputs.
                 retrieved.open('model.pb', mode='rb'))
        self.ctx.model = SinglefileData(file=model)
        self.out('model.pb', self.ctx.model.store())

    def write_results(self):
        os.chdir(self.inputs.resdir)
        with open(RESULT_NAME, 'a') as f:
            f.write(f'# Step: Deepmd Training, '
                    f'PK: {self.ctx.dpmd_workchain.pk}\n')
