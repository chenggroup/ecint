import os
from time import sleep
import multiprocessing
from ase.io import read
from ecint.workflow import *
from aiida.orm import StructureData
from aiida.engine import run_get_node, submit
from ecint.preprocessor.utils import notification_in_dingtalk, load_json
from aiida import load_profile

load_profile()


class ecint(object):
    def __init__(self, workflow, input_paras):
        self.workflow = workflow
        self.input_paras = input_paras
        # if structure_list has the same format
        if 'structures' in input_paras:
            if len(set([os.path.splitext(structure_file)[1] for structure_file in input_paras.get('structures')])) != 1:
                raise ValueError('Structure file in structures should have the same format')
        # setup self.structure or self.structures_list
        # for .xyz, you should set cell and pbc
        if (input_paras.get('structure') or input_paras.get('structures')[0]).endswith('.xyz'):
            if (input_paras.get('cell') is None) or (input_paras.get('pbc') is None):
                raise ValueError('You should set up cell and pbc')
            cell = input_paras.pop('cell')
            pbc = input_paras.pop('pbc')
            if 'structure' in input_paras:
                atoms = read(input_paras.get('structure'))
                atoms.set_cell(cell)
                atoms.set_pbc(pbc)
                structure = StructureData(ase=atoms)
                self.input_paras['structure'] = structure
            elif 'structures' in input_paras:
                structures = {}
                for i, structure_file in enumerate(input_paras.get('structures')):
                    atoms = read(structure_file)
                    atoms.set_cell(cell)
                    atoms.set_pbc(pbc)
                    structure = StructureData(ase=atoms)
                    structures.update({f'image_{i}': structure})
                self.input_paras['structures'] = structures
        else:
            if 'structure' in input_paras:
                atoms = read(input_paras.get('structure'))
                structure = StructureData(ase=atoms)
                self.input_paras['structure'] = structure
            elif 'structures' in input_paras:
                structures = {}
                for i, structure_file in enumerate(input_paras.get('structures')):
                    atoms = read(structure_file)
                    structure = StructureData(ase=atoms)
                    structures.update({f'image_{i}': structure})
                self.input_paras['structures'] = structures

    def run(self):
        process_dict, node = run_get_node(self.workflow, **self.input_paras)
        return node


def main():
    # workflow_name = eval(input('Select your workflow: '))
    # structures = input('Path of your structures(separate by space): ').split()
    # cell = eval(input('Matrix of cell, only need for .xyz[default=None]: ') or 'None')
    # pbc = eval(input('Array of pbc, only need for .xyz[default=None]: ') or 'None')
    # workdir = os.path.abspath(input('Directory of output files will be in[default=./]: '))
    # webhook = input('Your dingtalk webhook[default=None]: ') or None
    # input_paras = {'structures': structures, 'workdir': workdir}
    # if cell is not None:
    #     input_paras.update({'cell': cell})
    # if pbc is not None:
    #     input_paras.update({'pbc': pbc})
    # if workdir is not None:
    #     input_paras.update({'workdir': workdir})
    # TODO: make this part more general for other workflows
    user_input = load_json('ecint.json')
    workdir = os.path.abspath(user_input['workdir'])
    print(f'ECINT start in {os.getcwd()}')
    if not os.path.exists(workdir):
        os.mkdir(workdir)
    input_paras = {'structures': user_input['structures'], 'workdir': workdir}
    if user_input.get('cell') is not None:
        input_paras.update({'cell': user_input.get('cell')})
    if user_input.get('pbc') is not None:
        input_paras.update({'pbc': user_input.get('pbc')})
    e = ecint(eval(user_input['workflow']), input_paras)
    node = e.run()
    if input_paras.get('webhook') is not None:
        notification_in_dingtalk(input_paras['webhook'], node)
