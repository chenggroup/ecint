import re

from ecint.postprocessor.utils import parse_band_convergence_like_info


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
