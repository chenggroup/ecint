import numpy as np
from ase import Atoms
from ase.io import read, write

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


def get_series_file_name():
    # TODO: auto get aiida-pos-Replica_nr_{i}-1.xyz
    pass


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


def get_convergence_info_of_band(node):
    """
    check if BAND.out is convergent
    :param node:
    :return:
    """
    # get remote_path


def write_output_files():
    # split write output files step from workchain
    pass

