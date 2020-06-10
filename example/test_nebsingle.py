from ecint.workflow.units.base import NebSingleWorkChain
from ase.io import read
from aiida.orm import StructureData, List
from aiida.engine import run
from aiida import load_profile
load_profile()

structure_file_list = ['ethane_1_opt.xyz', 'ethane_s1.xyz', 'ethane_ts.xyz', 'ethane_s2.xyz']
structures = {}
for i, structure_file in enumerate(structure_file_list):
    atoms = read(structure_file)
    atoms.set_cell([12, 12, 12])
    structure = StructureData(ase=atoms)
    structures.update({f'image_{i}': structure})

input_paras = {'structures': structures}
run(NebSingleWorkChain, **input_paras)
