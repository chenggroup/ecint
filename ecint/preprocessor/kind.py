from abc import ABCMeta, abstractmethod

import yaml

_E_WITH_Q = {'H': '1', 'He': '2', 'Li': '3', 'Be': '4', 'B': '3', 'C': '4', 'N': '5', 'O': '6', 'F': '7', 'Ne': '8',
             'Na': '9', 'Mg': '2', 'Al': '3', 'Si': '4', 'P': '5', 'S': '6', 'Cl': '7', 'Ar': '8', 'K': '9', 'Ca': '10',
             'Sc': '11', 'Ti': '12', 'V': '13', 'Cr': '14', 'Mn': '15', 'Fe': '16', 'Co': '17', 'Ni': '18', 'Cu': '11',
             'Zn': '12', 'Ga': '3', 'Ge': '4', 'As': '5', 'Se': '6', 'Br': '7', 'Kr': '8',
             'Rb': '9', 'Sr': '10', 'Y': '11', 'Zr': '12', 'Nb': '13', 'Mo': '14', 'Tc': '15', 'Ru': '8', 'Rh': '9',
             'Pd': '18', 'Ag': '11', 'Cd': '12', 'In': '3', 'Sn': '4', 'Sb': '5', 'Te': '6', 'I': '7', 'Xe': '8',
             'Cs': '9', 'Ba': '10', 'La': '11', 'Hf': '12', 'Ta': '5', 'W': '6', 'Re': '7', 'Os': '8', 'Ir': '9',
             'Pt': '18', 'Au': '19', 'Hg': '12', 'Tl': '3', 'Pb': '4', 'Bi': '5', 'Po': '6', 'At': '7', 'Rn': '8'}


class BaseSets(metaclass=ABCMeta):
    def __init__(self, structure):
        self.structure = structure
        self.elements = self.structure.get_symbols_set()

    @property
    @abstractmethod
    def kind_section(self):
        pass


class SetsFromYaml(BaseSets):

    def __init__(self, structure, kind_section_config_path):
        super(SetsFromYaml, self).__init__(structure)
        self.kind_section_config = self.load_kind_section_config_file(kind_section_config_path)

    @property
    def kind_section(self):
        if self.kind_section_config.keys() == self.elements:
            kind_section_list = []
            for k, v in self.kind_section_config.items():
                one_kind_section = {'_': k}
                one_kind_section.update(v)
                kind_section_list.append(one_kind_section)
        else:
            raise ValueError('Elements in input structure and configuration file do not match')
        return kind_section_list

    @staticmethod
    def load_kind_section_config_file(kind_section_config_path):
        try:
            with open(kind_section_config_path, 'r') as f:
                kind_section_config = yaml.load(f)
        except IOError:
            print('Can not find file {}'.format(kind_section_config_path))
        return kind_section_config


class TZV2PSets(BaseSets):
    @property
    def kind_section(self):
        kind_section_list = []
        for e in self.elements:
            one_kind_section = {'_': e, 'BASIS_SET': 'TZV2P-GTH',
                                'POTENTIAL': 'GTH-BLYP-q{}'.format(_E_WITH_Q[e])}
            kind_section_list.append(one_kind_section)
        return kind_section_list


class DZVPSets(BaseSets):
    @property
    def kind_section(self):
        kind_section_list = []
        for e in self.elements:
            one_kind_section = {'_': e, 'BASIS_SET': 'DZVP-MOLOPT-SR-GTH',
                                'POTENTIAL': 'GTH-PBE-q{}'.format(_E_WITH_Q[e])}
            kind_section_list.append(one_kind_section)
        return kind_section_list
