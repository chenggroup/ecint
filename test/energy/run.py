from aiida.orm import Str
from aiida.engine import run_get_node
from energy import EnergyWorkChain
from aiida import load_profile
load_profile()

input_files = {'structure_file': Str('h2o.xyz'),
               'kind_section_file': Str('kind_section.yaml'),
               'machine_file': Str('machine.json')}

# submit workflow
# builder = EnergyWorkChain.get_builder()
submit_dict, submit_node = run_get_node(EnergyWorkChain, **input_files)
# node = submit(EnergyWorkChain, **input_files)
# get_result
"""
拿结果的类不宜放在 workchain 里（没法创建 workchain class 的实例），也许可以有另外的类包括工作流提交，拿结果之类的步骤
def get_result(self):
    node = self.ctx.workchain
    results = node.outputs.output_parameters.get_dict()
    value, units = results['energy'], results['energy_units']
    return f'energy: {value} {units}'
"""
