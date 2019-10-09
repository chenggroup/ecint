#!/usr/bin/env runaiida

import json
import sys

import numpy as np
from aiida.common import NotExistent
from aiida.engine import submit
from aiida.orm import (Code, Dict, StructureData)
from aiida_cp2k.workchains import Cp2kBaseWorkChain
from ase.io import read

from ecint.preprocessor.kind import DzvpSets


def main(codelabel, structure_path, scale_range, need_change):
    """Run simple DFT calculation through a workchain"""
    try:
        code = Code.get_from_string(codelabel)
    except NotExistent:
        print("The code '{}' does not exist".format(codelabel))
        sys.exit(1)

    print("Testing CP2K ENERGY on H2O (DFT) through a workchain...")

    # structure
    atoms = read(structure_path)
    (a, b, c, alpha, beta, gamma) = atoms.get_cell_lengths_and_angles()
    k = np.ones([len(scale_range), 3])

    # need input check
    if 'a' in need_change:
        k.T[0] = scale_range
    if 'b' in need_change:
        k.T[1] = scale_range
    if 'c' in need_change:
        k.T[2] = scale_range

    atoms_list = []
    for k_a, k_b, k_c in k:
        atoms.set_cell([k_a * a, k_b * b, k_c * c, alpha, beta, gamma], scale_atoms=True)
        atoms_list.append(atoms.copy())

    # options
    options = {
        "resources": {
            "tot_num_mpiprocs": 24,
        },
        "max_wallclock_seconds": 1 * 60 * 60,
        "queue_name": "medium"
    }

    # base json
    with open('energy.json', 'r') as f:
        params = json.load(f)
    # parameters
    for s in atoms_list:
        structure = StructureData(ase=s)
        kindlist = DzvpSets(structure=structure).kind_section
        parameter_dict = params
        parameter_dict['FORCE_EVAL']['SUBSYS']['KIND'] = kindlist
        parameters = Dict(dict=parameter_dict)
        inputs = {
            'cp2k': {
                'structure': structure,
                'parameters': parameters,
                'code': code,
                'metadata': {
                    'options': options,
                }
            }
        }
        print("Submitted calculation...")
        submit(Cp2kBaseWorkChain, **inputs)


if __name__ == '__main__':
    main('cp2k@chenglab51', '~/aiida/h2o.xyz', np.arange(0.1, 0.5, 0.2),
         ['a', 'b'])  # pylint: disable=no-value-for-parameter
