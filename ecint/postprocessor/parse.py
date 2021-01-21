import re

import numpy as np


def parse_band_convergence_like_info(str_info):
    band_convergence_like_info_dict = {
        'step_value': float(re.search(r'=(.*)\[', str_info).group(1)),
        'convergence_criteria': float(re.search(r'\[(.*)\]',
                                                str_info).group(1)),
        'is_converged': re.search(r'(YES|NO)', str_info).group(0)
    }
    return band_convergence_like_info_dict


def parse_model_devi_index(filename, skip_images,
                           force_low_limit, force_high_limit,
                           energy_low_limit, energy_high_limit):
    model_devi_array = np.loadtxt(filename, usecols=[0, 4, 1])
    valid_model_devi = model_devi_array[model_devi_array[:, 0] >= skip_images]
    traj_step, force_devi, energy_devi = valid_model_devi.T
    # TODO: now only consider cluster_cutoff is None, to consider
    #  other situations when needed
    candidate_index = np.union1d(np.argwhere((force_low_limit <= force_devi) &
                                             (force_devi < force_high_limit)),
                                 np.argwhere((energy_low_limit <= energy_devi) &
                                             (energy_devi < energy_high_limit)))
    failed_index = np.union1d(np.argwhere(force_devi >= force_high_limit),
                              np.argwhere(energy_devi >= energy_high_limit))
    accurate_index = np.intersect1d(np.argwhere(force_devi < force_low_limit),
                                    np.argwhere(energy_devi < energy_low_limit))
    traj_step = traj_step.astype(int)
    return {'candidate': traj_step[candidate_index],
            'failed': traj_step[failed_index],
            'accurate': traj_step[accurate_index]}
