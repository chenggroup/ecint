import json
import re

import numpy as np
import requests
from ase import Atoms
from ase.io import read, write

from ecint.postprocessor.parse import parse_band_convergence_like_info

AU2EV = 27.2113838565563

PROJECT_NAME = 'aiida'
TRAJ_NAME = f'{PROJECT_NAME}-pos-1.xyz'
REPLICA_NAME = f'{PROJECT_NAME}-Replica_data_for_energy_curve.xyz'
MAX_ENERGY_NAME = f'max_energy_structure.xyz'


def get_last_frame(traj_file=TRAJ_NAME, format='xyz', cell=None, pbc=None):
    """
    :param traj_file: file or filelike
    :return: atoms
    """
    last_frame = read(traj_file, index='-1', format=format)
    last_frame.set_cell(cell)
    last_frame.set_pbc(pbc)
    return last_frame


def get_traj_for_energy_curve(replica_traj_list, write_name=REPLICA_NAME):
    """
    :param replica_traj_list: atoms list (traj list) or file/filelike list
    :param write_name:
    do not write output file, set write_name=None or ''
    """
    if isinstance(replica_traj_list[0], str):
        traj_for_energy_curve = [get_last_frame(replica_traj_file) for replica_traj_file in replica_traj_list]
    elif isinstance(replica_traj_list[0], Atoms):
        traj_for_energy_curve = [replica_traj for replica_traj in replica_traj_list]
    else:
        raise ValueError('`replica_traj_list` need be list of atoms or file/filelike')
    if write_name:
        write(write_name, traj_for_energy_curve)
    return traj_for_energy_curve


def get_max_energy_frame(traj_file=REPLICA_NAME, write_name=MAX_ENERGY_NAME, cell=None, pbc=False):
    traj = read(traj_file, index=':')
    energy_list = np.array([atoms.info['E'] for atoms in traj])
    atoms_max_energy = traj[energy_list.argmax()]
    atoms_max_energy.set_cell(cell)
    atoms_max_energy.set_pbc(pbc)
    if write_name:
        atoms_max_energy.write(write_name)
    return atoms_max_energy


def write_output_files():
    # split write output files step from workchain
    pass


def write_xyz_from_structure(structure, output_file):
    """Write output xyz file for StructureData with energy

    Args:
        structure (aiida.orm.StructureData): StructureData with attribute energy
        output_file (str): output file name, *.xyz

    Returns:
        None

    """
    atoms = structure.get_ase()
    atoms.info.update({'E': f'{structure.get_attribute("energy")} eV'})
    atoms.write(output_file)


def write_xyz_from_trajectory(trajectory, output_file):
    """Write output xyz file for TrajectoryData with energy

    Args:
        trajectory (aiida.orm.TrajectoryData): TrajectoryData with array energy
        output_file (str): output file name, *.xyz

    Returns:
        None

    """
    energy_array = trajectory.get_array('energy')
    traj = []
    for structure_index in trajectory.get_stepids():
        structure = trajectory.get_step_structure(structure_index)
        energy = energy_array[structure_index]
        atoms = structure.get_ase()
        atoms.info.update({'i': structure_index, 'E': f'{energy} eV'})
        traj.append(atoms)
    write(output_file, traj)


def notification_in_dingtalk(webhook, node):
    """Send messages to dingtalk

    Args:
        webhook (str): url webhook of dingtalk robot
        node (aiida.orm.ProcessNode): object returned by running or submitting workchain,
                                      at ecint WorkChain or SingleWorkChain level

    Returns:
        dict: response information after post

    """
    headers = {'Content-Type': 'application/json'}
    title = 'Job Info'
    structure = getattr(node.inputs, next(filter(lambda x: re.match(r'structure.*', x), dir(node.inputs))))
    text = '## Job Info\n'
    text += 'Your job is over!\n'
    text += '>\n'
    text += f'> Job PK: **{node.pk}**\n'
    text += '>\n'
    text += f'> Job Chemical Formula: **{structure.get_formula()}**\n'
    text += '>\n'
    text += f'> Job Type: **{node.process_label}**\n'
    text += '>\n'
    text += f'> Job State: **{node.process_state.name}**\n'
    data = {'msgtype': 'markdown', 'markdown': {'title': title, 'text': text}}
    response = requests.post(url=webhook, headers=headers, data=json.dumps(data))
    return response


def get_convergence_info_of_band(band_file):
    """
    check if BAND.out is convergent
    :param band_file:
    :return:
    """
    with open(band_file) as f:
        band_info = f.read()
    rms_displacement = re.findall(r'RMS DISPLACEMENT.*', band_info)
    max_displacement = re.findall(r'MAX DISPLACEMENT', band_info)
    rms_force = re.findall(r'RMS FORCE', band_info)
    max_force = re.findall(r'MAX FORCE', band_info)

    def get_convergence_info_list(convergence_key):
        return [parse_band_convergence_like_info(info) for info in convergence_key]

    band_convergence_info = {
        'rms_displacement': get_convergence_info_list(rms_displacement),
        'max_displacement': get_convergence_info_list(max_displacement),
        'rms_force': get_convergence_info_list(rms_force),
        'max_force': get_convergence_info_list(max_force)
    }
    return band_convergence_info
