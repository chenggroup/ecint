from ecint.workflow.units.base import GeooptSingleWorkChain
from ase.io import read
from aiida.orm import StructureData
from aiida.engine import run
from aiida import load_profile
load_profile()

atoms = read('ethane.xyz')
atoms.set_cell([12, 12, 12])
structure = StructureData(ase=atoms)

input_paras = {'structure': structure}
run(GeooptSingleWorkChain, **input_paras)
