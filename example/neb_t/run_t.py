import os
from test_t import NebWorkChain
from aiida.engine import run_get_node, submit
from aiida import load_profile
load_profile()

print('START')
results_dir = 'results'
if not os.path.exists(results_dir):
    os.mkdir(results_dir)
os.chdir(results_dir)
print(f'Now in {os.getcwd()}')

input_files = {'structure_list': ['../is.xyz', '../Replica3.xyz', '../fs.xyz'],
               'kind_section_file': '../kind_section.yaml',
               'machine_file': '../machine.json'}

submit_dict, submit_node = run_get_node(NebWorkChain, input_files=input_files)
# add dingtalk notification
print('END')
