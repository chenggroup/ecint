# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
###############################################################################
# Copyright (c), The AiiDA-CP2K authors.                                      #
# SPDX-License-Identifier: MIT                                                #
# AiiDA-CP2K is hosted on GitHub at https://github.com/aiidateam/aiida-cp2k   #
# For further information on the license, see the LICENSE.txt file.           #
###############################################################################
"""Run simple DFT calculation."""

from time import sleep

from aiida import load_profile
from aiida.engine import run_get_node
from aiida.orm import (Code, Dict, StructureData)
from ase.build import molecule

load_profile()


def example_dft(cp2k_code):
    """Run simple DFT calculation."""

    print("Testing CP2K ENERGY on H2O (DFT)...")

    # Structure.
    atoms = molecule('H2O')
    atoms.center(vacuum=5)
    structure = StructureData(ase=atoms)

    # Parameters.
    parameters = Dict(
        dict={
            'FORCE_EVAL': {
                'METHOD': 'Quickstep',
                'DFT': {
                    'BASIS_SET_FILE_NAME': 'BASIS_MOLOPT',
                    'POTENTIAL_FILE_NAME': 'GTH_POTENTIALS',
                    'QS': {
                        'EPS_DEFAULT': 1.0e-12,
                        'WF_INTERPOLATION': 'ps',
                        'EXTRAPOLATION_ORDER': 3,
                    },
                    'MGRID': {
                        'NGRIDS': 4,
                        'CUTOFF': 280,
                        'REL_CUTOFF': 30,
                    },
                    'XC': {
                        'XC_FUNCTIONAL': {
                            '_': 'LDA',
                        },
                    },
                    'POISSON': {
                        'PERIODIC': 'none',
                        'PSOLVER': 'MT',
                    },
                },
                'SUBSYS': {
                    'KIND': [
                        {
                            '_': 'O',
                            'BASIS_SET': 'DZVP-MOLOPT-SR-GTH',
                            'POTENTIAL': 'GTH-LDA-q6'
                        },
                        {
                            '_': 'H',
                            'BASIS_SET': 'DZVP-MOLOPT-SR-GTH',
                            'POTENTIAL': 'GTH-LDA-q1'
                        },
                    ],
                },
            }
        })

    # Construct process builder.
    builder = cp2k_code.get_builder()
    builder.structure = structure
    builder.parameters = parameters
    builder.code = cp2k_code
    builder.metadata.options.resources = {
        'tot_num_mpiprocs': 24,
        "num_mpiprocs_per_machine": 24,
    }
    builder.metadata.options.max_wallclock_seconds = 1 * 3 * 60
    builder.metadata.options.custom_scheduler_commands = f'#BSUB -R "span[ptile=24]"'

    print("Submitted calculation...")
    process, node = run_get_node(builder)

    while not node.is_terminated:
        sleep(5)
    from curl_test import notification_in_dingtalk
    webhook = 'https://oapi.dingtalk.com/robot/send?access_token=a3cd7e35c31f149248a46053f51b11ad843cc50a975730e565cb3f0292f8e56b'
    notification_in_dingtalk(webhook, node)


if __name__ == '__main__':
    code = Code.get_from_string('cp2k@chenglab52')
    example_dft(code)
