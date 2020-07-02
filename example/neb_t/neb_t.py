import re
from os import chdir, getcwd
from os.path import join, isabs

from aiida.engine import WorkChain
from aiida.orm import List
from ase.io import read

from ecint.postprocessor import get_traj_for_energy_curve, get_max_energy_frame
from ecint.preprocessor import GeooptPreprocessor, NebPreprocessor, FrequencyPreprocessor
from ecint.preprocessor.input import GeooptInputSets, NebInputSets, FrequencyInputSets
from ecint.preprocessor.utils import load_json, load_machine, check_neb
from ecint.workflow.units import CONFIG_DIR


class NebWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super(NebWorkChain, cls).define(spec)
        # TODO: use more parameters,
        #  use split kind_section, machine config for geoopt, neb, freq
        spec.input('workdir', valid_type=str, non_db=True)
        spec.input('input_files.structure_list', valid_type=List)  # List of StructureData
        spec.input('input_files.machine_file', valid_type=str, default='AutoMode', required=False, non_db=True)
        spec.input('input_files.geoopt_config_file', valid_type=str, default=join(CONFIG_DIR, 'geoopt.json'),
                   required=False, non_db=True)
        spec.input('input_files.neb_config_file', valid_type=str, default=join(CONFIG_DIR, 'neb.json'),
                   required=False, non_db=True)
        spec.input('input_files.frequency_config_file', valid_type=str, default=join(CONFIG_DIR, 'frequency.json'),
                   required=False, non_db=True)
        spec.input('input_files.kind_section_file', valid_type=str, default='DZVPBLYP', required=False, non_db=True)

        # TODO: set/change name of Cp2kBaseWorkChain(in verdi process list) to Geoopt, Neb, Freq
        #       maybe change the `process_label` (use node.set_process_label())
        spec.outline(
            cls.goto_workdir,
            cls.check_config,
            cls.check_input_structure_files,
            # cls.prepare_atoms,
            cls.submit_geoopt,
            cls.inspect_geoopt,
            cls.submit_neb,
            cls.inspect_neb,
            cls.get_energy_curve_data,
            cls.submit_frequency,
            cls.inspect_frequency,
            cls.get_frequency_data,
        )

    def goto_workdir(self):
        # workdir should be set, otherwise the workchain will work on ecint installation directory
        workdir = self.inputs.workdir
        if isabs(workdir):
            chdir(workdir)
            self.report(f'Work in {getcwd()}')
        else:
            raise ValueError('Workdir need be a absolute path')

    def check_config(self):
        self.ctx.neb_config = load_json(self.inputs.input_files.neb_config_file)
        # nproc_rep = self.ctx.neb_config['MOTION']['BAND'].get('NPROC_REP')
        number_of_replica = self.ctx.neb_config['MOTION']['BAND']['NUMBER_OF_REPLICA']
        if self.inputs.input_files.machine_file == 'AutoMode':
            _default_machine = {'code@computer': 'cp2k@aiida_test', 'nnode': number_of_replica, 'queue': 'large'}
            self.ctx.machine = load_machine(_default_machine)
        else:
            self.ctx.machine = load_machine(self.inputs.input_files.machine_file)
        check_neb(self.ctx.neb_config, self.ctx.machine)

    def check_input_structure_files(self):
        if len(self.inputs.input_files.structure_list) < 2:
            raise ValueError('Number of the structures in structure_list should be at least 2')

    # def prepare_atoms(self):
    #     # TODO: make pbc can change config.json
    #     # TODO: add constrains
    #     atoms = read(self.inputs.input_files.structure_list[0])
    #     self.ctx.cell = atoms.cell
    #     self.ctx.pbc = atoms.pbc

    def submit_geoopt(self):
        reactant_structure = self.inputs.input_files.structure_list[0]
        product_structure = self.inputs.input_files.structure_list[-1]
        # submit workchains for reactant and product
        for i, structure in enumerate([reactant_structure, product_structure]):
            inputclass = GeooptInputSets(structure, config=self.inputs.input_files.geoopt_config_file,
                                         kind_section_config=self.inputs.input_files.kind_section_file)
            pre = GeooptPreprocessor(inputclass, self.ctx.machine)
            builder = pre.builder
            node = self.submit(builder)
            if i == 0:  # first structure in list is reactant
                self.to_context(geoopt_reactant_workchain=node)
            elif i == 1:  # second structure in list is reactant
                self.to_context(geoopt_product_workchain=node)

    def inspect_geoopt(self):
        # TODO: use return exitcode instead of assert
        # geoopt_is_finished_ok = []
        for node in [self.ctx.geoopt_reactant_workchain, self.ctx.geoopt_product_workchain]:
            assert node.is_finished_ok
        # return all(geoopt_is_finished_ok)

    def submit_neb(self):
        reactant_geoopt = self.ctx.geoopt_reactant_workchain.outputs.output_structure
        product_geoopt = self.ctx.geoopt_product_workchain.outputs.output_structure
        # {'filename': StructureData}, StructureData which need be submitted to remote
        replica_dict = {'reactant_geoopt': reactant_geoopt, 'product_geoopt': product_geoopt}

        # pre setup inputclass
        inputclass = NebInputSets(reactant_geoopt, config=self.inputs.input_files.neb_config_file,
                                  kind_section_config=self.inputs.input_files.kind_section_file)
        # add reactant and product, then submit them to remote
        inputclass.add_config({'MOTION': {'BAND': {'REPLICA': [{'COORD_FILE_NAME': 'reactant_geoopt.xyz'},
                                                               {'COORD_FILE_NAME': 'product_geoopt.xyz'}]}}})
        # add other input structures, then submit them to remote
        other_images = self.inputs.input_files.structure_list[1:-1]
        if other_images:
            for i, image in enumerate(other_images):
                replica_dict.update({f'image_{i}': image})
                inputclass.add_config({'MOTION': {'BAND': {'REPLICA': [{'COORD_FILE_NAME': f'image_{i}.xyz'}]}}})

        pre = NebPreprocessor(inputclass, self.ctx.machine)
        builder = pre.builder
        builder.cp2k.file = replica_dict
        node = self.submit(builder)
        self.to_context(neb_workchain=node)

    def inspect_neb(self):
        # TODO: check convergence in BAND.out,
        #       test a job which walltime is reached
        node = self.ctx.neb_workchain
        assert node.is_finished_ok
        # return node.is_finished_ok

    def get_energy_curve_data(self):
        self.ctx.energy_curve_file_name = 'Replica_data_for_energy_curve.xyz'
        node = self.ctx.neb_workchain
        # get list of `Atoms`
        replica_traj_list = []
        number_of_replica = self.ctx.neb_config['MOTION']['BAND']['NUMBER_OF_REPLICA']
        for i in range(1, number_of_replica + 1):
            # warning: if project name is not 'aiida', this part will fail
            with node.outputs.retrieved.open(f'aiida-pos-Replica_nr_{i}-1.xyz') as replica_file:
                replica_traj_list.append(read(replica_file, index='-1', format='xyz'))
        get_traj_for_energy_curve(replica_traj_list, write_name=self.ctx.energy_curve_file_name)

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
