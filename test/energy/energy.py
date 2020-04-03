from aiida.engine import WorkChain, ToContext, ExitCode, if_
from aiida.orm import Str
from ase.io import read

from ecint.preprocessor import EnergyPreprocessor
from ecint.preprocessor.input import EnergyInputSets


class EnergyWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super(EnergyWorkChain, cls).define(spec)
        spec.input('structure_file', valid_type=Str)
        spec.input('config_file', valid_type=Str, required=False)
        spec.input('kind_section_file', valid_type=Str, required=False)
        spec.input('machine_file', valid_type=Str)

        spec.outline(
            cls.submit_workchain,
            if_(cls.inspect_workchain)(
                cls.get_result,
            )
        )

    def submit_workchain(self):
        atoms = read(self.inputs.structure_file.value)
        inputclass = EnergyInputSets(atoms, config='energy.json',  # use default energy config
                                     kind_section_config=self.inputs.kind_section_file.value)
        pre = EnergyPreprocessor(inputclass)
        pre.load_machine_from_json(self.inputs.machine_file.value)
        builder = pre.builder
        node = self.submit(builder)
        return ToContext(workchain=node)

    def inspect_workchain(self):
        return self.ctx.workchain.is_finished_ok

    def get_result(self):
        node = self.ctx.workchain
        results = node.outputs.output_parameters.get_dict()
        value, units = results['energy'], results['energy_units']
        with open('results.txt', 'w') as f:
            f.writelines([f'pk: {node.pk}\n', f'energy: {value} {units}'])
        return ExitCode(0, 'Generate results.txt')
