from ase.io import read
from aiida.orm import StructureData
import json


def path2structure(structure_path):
    """
    :param structure_path:
    :return: StructureData
    """
    atoms = read(structure_path)
    structure = StructureData(ase=atoms)
    return structure


def load_json(json_path):
    with open(json_path) as f:
        d = json.load(f)
    return d


def inp2json(cp2k_input):
    # TODO: need edit, parse cp2k input file to json format
    pass
