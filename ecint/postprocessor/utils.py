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

    Args:
        traj_file (str): file or filelike obj
        format (str): supported format of `ase`
        cell (list): cell of `ase`
        pbc (bool or List[bool]): pbc of `ase`

    Returns:
        ase.Atoms: last frame of input traj_file

    """

    last_frame = read(traj_file, index='-1', format=format)
    last_frame.set_cell(cell)
    last_frame.set_pbc(pbc)
    return last_frame


def get_traj_for_energy_path(replica_traj_list, write_name=REPLICA_NAME):
    """Could generate a trajectory file

    Args:
        replica_traj_list (list): atoms list (trajectory) or file/filelike list
        write_name (str): name of output trajectory file,
            if do not want to write output file, set write_name=None or ''

    Returns:
        ase.Atoms: trajectory of energy path

    """

    if isinstance(replica_traj_list[0], str):
        traj_for_energy_path = [get_last_frame(replica_traj_file) for
                                replica_traj_file in replica_traj_list]
    elif isinstance(replica_traj_list[0], Atoms):
        traj_for_energy_path = [replica_traj for replica_traj in
                                replica_traj_list]
    else:
        raise ValueError('`replica_traj_list` need be '
                         'list of atoms or file/filelike')
    if write_name:
        write(write_name, traj_for_energy_path)
    return traj_for_energy_path


def get_max_energy_frame(traj_file=REPLICA_NAME, write_name=MAX_ENERGY_NAME,
                         cell=None, pbc=False):
    """Could generate a structure file

    Args:
        traj_file (str): structure file with many frames
        write_name (str): name of the max frame file,
            if do not want to write output file, set write_name=None or ''
        cell (list): cell of `ase`
        pbc (bool, List[bool]): pbc of `ase`

    Returns:

    """
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
        node (aiida.orm.ProcessNode):
            object returned by running or submitting workchain,
            at ecint WorkChain or SingleWorkChain level

    Returns:
        dict: response information after post

    """
    headers = {'Content-Type': 'application/json'}
    title = 'Job Info'
    # get structure
    structure = None
    try:
        structure = getattr(node.inputs,
                            next(filter(lambda x: re.match(r'structure.*', x),
                                        dir(node.inputs))))
    except:
        pass
    text = '## Job Info\n'
    text += 'Your job is over!\n'
    text += '>\n'
    text += f'> Job PK: **{node.pk}**\n'
    text += '>\n'
    if structure:
        text += f'> Job Chemical Formula: **{structure.get_formula()}**\n'
    text += '>\n'
    text += f'> Job Type: **{node.process_label}**\n'
    text += '>\n'
    text += f'> Job State: **{node.process_state.name}**\n'
    data = {'msgtype': 'markdown', 'markdown': {'title': title, 'text': text}}
    response = requests.post(url=webhook, headers=headers,
                             data=json.dumps(data))
    return response


def get_convergence_info_of_band(band_file):
    """Check if BAND.out is convergent

    Args:
        band_file (str): name of band file

    Returns:
        dict: convergence information of band

    """

    with open(band_file) as f:
        band_info = f.read()
    rms_displacement = re.findall(r'RMS DISPLACEMENT.*', band_info)
    max_displacement = re.findall(r'MAX DISPLACEMENT', band_info)
    rms_force = re.findall(r'RMS FORCE', band_info)
    max_force = re.findall(r'MAX FORCE', band_info)

    def get_convergence_info_list(convergence_key):
        return [parse_band_convergence_like_info(info) for info in
                convergence_key]

    band_convergence_info = {
        'rms_displacement': get_convergence_info_list(rms_displacement),
        'max_displacement': get_convergence_info_list(max_displacement),
        'rms_force': get_convergence_info_list(rms_force),
        'max_force': get_convergence_info_list(max_force)
    }
    return band_convergence_info
