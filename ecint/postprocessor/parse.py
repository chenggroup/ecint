import re


def parse_band_convergence_like_info(str_info):
    band_convergence_like_info_dict = {'step_value': float(re.search(r'=(.*)\[', str_info).group(1)),
                                       'convergence_criteria': float(re.search(r'\[(.*)\]', str_info).group(1)),
                                       'is_converged': re.search(r'(YES|NO)', str_info).group(0)}
    return band_convergence_like_info_dict
