from ase.io import read, write
import numpy as np

PROJECT_NAME = 'aiida'
TRAJ_NAME = f'{PROJECT_NAME}-pos-1.xyz'
REPLICA_NAME = f''


def get_last_frame(traj=TRAJ_NAME):
    """
    :param traj: is xyz file
    :return: xyz file
    """
    last_frame = read(traj, '-1')
    return last_frame


def get_series_file_name():
    pass


def get_traj_for_energy_curve(traj_list, write_xyz=True):
    traj_list_for_energy_curve = []
    for replica_traj in traj_list:
        last_frame = get_last_frame(replica_traj)
        traj_list_for_energy_curve.append(last_frame)
    if write_xyz:
        write(REPLICA_NAME, traj_list_for_energy_curve)
    return traj_list_for_energy_curve


def get_max_energy_frame(traj_list):
    energy_list = np.array([traj.info['E'] for traj in traj_list])
    energy_list.argmax()
