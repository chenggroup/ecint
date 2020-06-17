import os
from ecint.workflow.neb import NebWorkChain
from ase.io import read
from aiida.orm import StructureData, List
from aiida.engine import run, submit
from aiida import load_profile
load_profile()


def submit_workchain(input_paras, cell=None, pbc=None):
    if input_paras['structures_list'][0].endswith('.xyz'):
        if (cell is None) or (pbc is None):
            raise ValueError('You should set up cell and pbc')
    structures = {}
    for i, structure_file in enumerate(input_paras.pop('structures_list')):
        atoms = read(structure_file)
        atoms.set_cell(cell)
        atoms.set_pbc(pbc)
        structure = StructureData(ase=atoms)
        structures.update({f'image_{i}': structure})
    submit(NebWorkChain, structures=structures, **input_paras)


if __name__ == '__main__':
    results_dir = 'results'
    structure_file_list = ['ethane_1_opt.xyz', 'ethane_s1.xyz', 'ethane_ts.xyz', 'ethane_s2.xyz']
    cell = [12, 12, 12]
    pbc = False

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    input_paras = {'structures_list': structure_file_list, 'workdir': os.path.abspath(results_dir)}
    print('START')
    print(f'Now in {os.getcwd()}')
    print(f'Work directory is {os.path.abspath(results_dir)}')
    submit_workchain(input_paras, cell, pbc)
