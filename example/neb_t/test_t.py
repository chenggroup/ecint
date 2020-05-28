from os.path import abspath, join
from warnings import warn
import re

from aiida.engine import WorkChain, append_
from aiida.orm import SinglefileData
from ase.io import read, write

from ecint.workflow.units import CONFIG_DIR
from ecint.postprocessor import get_last_frame, get_traj_for_energy_curve, get_max_energy_frame
from ecint.preprocessor import GeooptPreprocessor, NebPreprocessor, FrequencyPreprocessor
from ecint.preprocessor.utils import load_json, load_machine, check_neb
from ecint.preprocessor.input import GeooptInputSets, NebInputSets, FrequencyInputSets


class NebWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super(NebWorkChain, cls).define(spec)
        # TODO: use more parameters,
        #  use split kind_section, machine config for geoopt, neb, freq
        spec.input('input_files.structure_list', valid_type=list, non_db=True)  # TODO: make structure_list[atoms, ...]
        spec.input('input_files.machine_file', valid_type=str, default='AutoMode', required=False, non_db=True)
        spec.input('input_files.geoopt_config_file', valid_type=str, default=join(CONFIG_DIR, 'geoopt.json'),
                   required=False, non_db=True)
        spec.input('input_files.neb_config_file', valid_type=str, default=join(CONFIG_DIR, 'neb.json'), required=False,
                   non_db=True)
        spec.input('input_files.frequency_config_file', valid_type=str, default=join(CONFIG_DIR, 'frequency.json'),
                   required=False, non_db=True)
        spec.input('input_files.kind_section_file', valid_type=str, default='DZVPBLYP', required=False, non_db=True)

        spec.outline(
            cls.check_config,
            cls.prepare_atoms,
            cls.submit_geoopt,
            cls.inspect_geoopt,
            cls.submit_neb,
            cls.inspect_neb,
            cls.get_energy_curve_data,
            cls.submit_frequency,
            cls.inspect_frequency,
            cls.get_frequency_data,
        )

    def check_config(self):
        self.ctx.neb_config = load_json(self.inputs.input_files.neb_config_file)
        # nproc_rep = self.ctx.neb_config['MOTION']['BAND'].get('NPROC_REP')
        number_of_replica = self.ctx.neb_config['MOTION']['BAND']['NUMBER_OF_REPLICA']
        if self.inputs.input_files.machine_file == 'AutoMode':
            _default_machine = {'code@computer': 'cp2k@chenglab51', 'nnode': number_of_replica, 'queue': 'large'}
            self.ctx.machine = load_machine(_default_machine)
        else:
            self.ctx.machine = load_machine(self.inputs.input_files.machine_file)
        check_neb(self.ctx.neb_config, self.ctx.machine)

    def prepare_atoms(self):
        # TODO: make pbc can change config.json
        atoms = read(self.inputs.input_files.structure_list[0])
        self.ctx.cell = atoms.cell
        self.ctx.pbc = atoms.pbc

    def submit_geoopt(self):
        for structure_file in self.inputs.input_files.structure_list:
            atoms = read(structure_file)
            atoms.set_cell(self.ctx.cell)
            atoms.set_pbc(self.ctx.pbc)
            inputclass = GeooptInputSets(atoms, config=self.inputs.input_files.geoopt_config_file,
                                         kind_section_config=self.inputs.input_files.kind_section_file)
            pre = GeooptPreprocessor(inputclass, self.ctx.machine)
            builder = pre.builder
            node = self.submit(builder)
            self.to_context(geoopt_workchain=append_(node))

    def inspect_geoopt(self):
        # TODO: use return exitcode instead of assert
        # geoopt_is_finished_ok = []
        for node in self.ctx.geoopt_workchain:
            assert node.is_finished_ok
        # return all(geoopt_is_finished_ok)

    def submit_neb(self):
        atoms_list = []
        for node in self.ctx.geoopt_workchain:
            with node.outputs.retrieved.open('aiida-pos-1.xyz') as structure_file:
                atoms_list.append(get_last_frame(structure_file, format='xyz', cell=self.ctx.cell, pbc=self.ctx.pbc))
        # pre setup inputclass
        inputclass = NebInputSets(atoms_list[0], config=self.inputs.input_files.neb_config_file,
                                  kind_section_config=self.inputs.input_files.kind_section_file)
        # setup replica in cp2k
        replica_dict = {}
        for replica_index, atoms in enumerate(atoms_list):
            replica_name = f'image_{replica_index}.xyz'
            atoms.write(replica_name)
            replica_dict.update({f'image_{replica_index}': SinglefileData(file=replica_name)})
            inputclass.add_config({'MOTION': {'BAND': {'REPLICA': [{'COORD_FILE_NAME': replica_name}]}}})

        pre = NebPreprocessor(inputclass, self.ctx.machine)
        builder = pre.builder
        builder.cp2k.file = replica_dict
        node = self.submit(builder)
        self.to_context(neb_workchain=node)

    def inspect_neb(self):
        # TODO: check convergence in BAND.out
        node = self.ctx.neb_workchain
        assert node.is_finished_ok
        # return node.is_finished_ok

    def get_energy_curve_data(self):
        self.ctx.energy_curve_file_name = 'Replica_data_for_energy_curve.xyz'
        node = self.ctx.neb_workchain
        # TODO: use re.search to match replica filename
        replica_last_frame_list = []
        for i in range(1, 7):
            with node.outputs.retrieved.open(f'aiida-pos-Replica_nr_{i}-1.xyz') as replica_file:
                replica_last_frame = get_last_frame(replica_file, format='xyz')
                replica_last_frame_list.append(replica_last_frame)
        write(self.ctx.energy_curve_file_name, replica_last_frame_list)

    def submit_frequency(self):
        self.ctx.transition_state_file_name = 'Transition_State.xyz'
        get_max_energy_frame(traj_file=self.ctx.energy_curve_file_name, write_name=self.ctx.transition_state_file_name,
                             cell=self.ctx.cell, pbc=self.ctx.pbc)
        atoms = read(self.ctx.transition_state_file_name)
        inputclass = FrequencyInputSets(atoms, config=self.inputs.input_files.frequency_config_file,
                                        kind_section_config=self.inputs.input_files.kind_section_file)
        pre = FrequencyPreprocessor(inputclass, self.ctx.machine)
        builder = pre.builder
        node = self.submit(builder)
        self.to_context(frequency_workchain=node)

    def inspect_frequency(self):
        node = self.ctx.frequency_workchain
        assert node.is_finished_ok
        # return node.is_finished_ok

    def get_frequency_data(self):
        self.ctx.frequency_data_file_name = 'frequency_data.txt'
        node = self.ctx.frequency_workchain
        output_content = node.outputs.retrieved.get_object_content('aiida.out')
        frequency_data = "\n".join(re.findall(r'VIB\|Frequency.*', output_content))
        with open(self.ctx.frequency_data_file_name, 'w') as frequency_data_file:
            frequency_data_file.write(frequency_data)
