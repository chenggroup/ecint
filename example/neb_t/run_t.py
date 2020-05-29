import os
from time import sleep
from ecint.workflow import NebWorkChain
from ecint.preprocessor.utils import notification_in_dingtalk
from aiida.engine import run_get_node, submit
from aiida import load_profile

load_profile()

# if you want to use `submit` to instead `run`, the abspath should be use
results_dir = 'results'
input_paras = {'workdir': os.path.abspath(results_dir),
               'input_files': {'structure_list': ['../is.xyz', '../Replica3.xyz', '../fs.xyz'],
                               'kind_section_file': '../kind_section.yaml',
                               'machine_file': '../machine.json'}}
webhook = 'https://oapi.dingtalk.com/robot/send?access_token=a3cd7e35c31f149248a46053f51b11ad843cc50a975730e565cb3f0292f8e56b'

print('START')
if not os.path.exists(results_dir):
    os.mkdir(results_dir)
os.chdir(results_dir)
print(f'Now in {os.getcwd()}')

node = submit(NebWorkChain, **input_paras)
# add dingtalk notification
while not node.is_terminated:
    sleep(5*60)
notification_in_dingtalk(webhook, node)
print('END')
