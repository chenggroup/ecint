import json

from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida_cp2k.calculations import Cp2kCalculation
from ase.io import read

from ecint.preprocessor.kind import DzvpSets


class SinglePointEnergy(Cp2kCalculation):

    def load_params(json_path):
        with open(json_path, 'r') as f:
            params = json.load(f)
        return params

    def get_structure(cif_path):
        atoms = read(cif_path)
        structure = StructureData(ase=atoms)
        return structure

    def kindlist(structure):
        sets = DzvpSets(structure=structure)
        return sets.kind_section

    def get_pw_parameters(params, kindlist):
        parameter_dict = params
        parameter_dict['FORCE_EVAL']['SUBSYS']['KIND'] = kindlist
        parameters = ParameterData(dict=parameter_dict)
        return parameters
