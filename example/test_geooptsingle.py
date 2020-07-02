from aiida import load_profile
from aiida.engine import run
from aiida.orm import StructureData
from ase.io import read

from ecint.workflow.units.base import GeooptSingleWorkChain

load_profile()

atoms = read('ethane.xyz')
atoms.set_cell([12, 12, 12])
structure = StructureData(ase=atoms)

input_paras = {'structure': structure}
run(GeooptSingleWorkChain, **input_paras)
