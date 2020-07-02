from aiida import load_profile
from aiida.engine import run
from aiida.orm import load_node

from ecint.workflow.units.base import FrequencySingleWorkChain

load_profile()

node = load_node(1648)
ts_structure = node.outputs.transition_state

input_paras = {'structure': ts_structure}
run(FrequencySingleWorkChain, **input_paras)
