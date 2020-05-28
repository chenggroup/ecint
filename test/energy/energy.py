from aiida.engine import WorkChain, ToContext, ExitCode
from aiida.orm import Dict
from ase.io import read

from ecint.preprocessor.input import InputSetsFromFile
from ecint.preprocessor import LSFPreprocessor


class EnergyInputSets(InputSetsFromFile):
    def __init__(self, structure, config='energy.json', kind_section_config='DZVPBLYP'):
        super(EnergyInputSets, self).__init__(structure, config, kind_section_config)


class EnergyPreprocessor(LSFPreprocessor):
    @property
    def builder(self):
        builder = super(EnergyPreprocessor, self).builder
        builder.settings = Dict(dict={'additional_retrieve_list': ["*-pos-1.xyz"]})
        return builder


class EnergyWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super(EnergyWorkChain, cls).define(spec)
        spec.input('input_files.structure_file', valid_type=str, non_db=True)
        spec.input('input_files.config_file', valid_type=str, default='energy.json', required=False, non_db=True)
        spec.input('input_files.kind_section_file', valid_type=str, default='DZVPBLYP', required=False, non_db=True)
        spec.input('input_files.machine_file', valid_type=str, default='machine.json', non_db=True)
        spec.input('parameters', required=False)

        spec.outline(
            cls.submit_workchain,
            cls.inspect_workchain,
            cls.get_result,
        )

    def submit_workchain(self):
        atoms = read(self.inputs.input_files.structure_file)
        inputclass = EnergyInputSets(atoms, config=self.inputs.input_files.config_file,  # use default energy config
                                     kind_section_config=self.inputs.input_files.kind_section_file)
        pre = EnergyPreprocessor(inputclass)
        pre.load_machine_from_json(self.inputs.input_files.machine_file)
        builder = pre.builder
        node = self.submit(builder)
        return ToContext(workchain=node)

    def inspect_workchain(self):
        # TODO: add exitcode for some special cases
        assert self.ctx.workchain.is_finished_ok

    def get_result(self):
        node = self.ctx.workchain
        results = node.outputs.output_parameters.get_dict()
        value, units = results['energy'], results['energy_units']
        with open('results.txt', 'w') as f:
            f.writelines([f'pk: {node.pk}\n', f'energy: {value} {units}'])
        return ExitCode(0, 'Generate results.txt')
