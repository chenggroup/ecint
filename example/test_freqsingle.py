from ecint.workflow.units.base import FrequencySingleWorkChain
from ase.io import read
from aiida.orm import StructureData, List, load_node
from aiida.engine import run
from aiida import load_profile
load_profile()

node = load_node(1648)
ts_structure = node.outputs.transition_state

input_paras = {'structure': ts_structure}
run(FrequencySingleWorkChain, **input_paras)
