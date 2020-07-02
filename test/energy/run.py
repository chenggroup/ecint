import os

from aiida import load_profile
from aiida.engine import run
from energy import EnergyWorkChain

load_profile()

print('START')
results_dir = 'results'
if not os.path.exists(results_dir):
    os.mkdir(results_dir)
os.chdir(results_dir)
print(f'Now in {os.getcwd()}')

input_files = {'structure_file': '../h2o.xyz',
               'kind_section_file': '../kind_section.yaml',
               'machine_file': '../machine.json'}
print('SUBMITTING...')
run(EnergyWorkChain, input_files=input_files)  # TODO: add entry to aiida
print('END')
# 记得把包加入 PYTHONPATH 环境变量里
# 后台运行的话用 nohup python run.py &
