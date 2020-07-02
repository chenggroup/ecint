from aiida import load_profile
from aiida.engine import run
from aiida.orm import StructureData
from ase.io import read

from ecint.workflow.neb import NebWorkChain

load_profile()

structure_file_list = ['ethane_1_opt.xyz', 'ethane_s1.xyz', 'ethane_ts.xyz', 'ethane_s2.xyz']
structures = {}
for i, structure_file in enumerate(structure_file_list):
    atoms = read(structure_file)
    atoms.set_cell([12, 12, 12])
    structure = StructureData(ase=atoms)
    structures.update({f'image_{i}': structure})

input_paras = {'structures': structures}
run(NebWorkChain, **input_paras)
