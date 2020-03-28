import os
from abc import ABCMeta, abstractmethod
from copy import deepcopy

from aiida_cp2k.utils import Cp2kInput

from ecint.preprocessor import load_json
from ecint.preprocessor.kind import *
from ecint.preprocessor.kind import BaseSets
from ecint.workflow.units import CONFIG_DIR


class BaseInput(metaclass=ABCMeta):
    @property
    @abstractmethod
    def input_sets(self):
        pass

    @abstractmethod
    def generate_cp2k_input_file(self):
        pass


class InputSetsFromFile(BaseInput):

    def __init__(self, structure, config_path, kind_section_config='DZVPSets'):
        """
        :param structure:
        :param config_path: base sets
        :param kind_section_config: kind_section sets, use path or TZV2PSets or DZVPSets
        """
        self.structure = structure
        self.config = load_json(config_path)
        # define self.kind_section
        if os.path.exists(kind_section_config):
            self.kind_section = SetsFromYaml(self.structure, kind_section_config).kind_section
        elif issubclass(eval(kind_section_config), BaseSets):
            __KindSectionSets = eval(kind_section_config)
            self.kind_section = __KindSectionSets(self.structure).kind_section
        else:
            raise (ValueError, 'Unexpected kind_section_config, please input a yaml file or a builtin set')

    @property
    def input_sets(self):
        _config = deepcopy(self.config)
        force_eval = _config["FORCE_EVAL"]
        if isinstance(force_eval, dict):
            force_eval["SUBSYS"]["KIND"] = self.kind_section
        elif isinstance(force_eval, list):
            for one_force_eval in force_eval:
                one_force_eval["SUBSYS"]["KIND"] = self.kind_section
        else:
            raise (ValueError, 'FORCE_EVAL section should be dict or list')
        return _config

    def generate_cp2k_input_file(self):
        from aiida_cp2k.calculations import Cp2kCalculation
        inp = Cp2kInput(self.input_sets)
        for i, letter in enumerate('ABC'):
            inp.add_keyword('FORCE_EVAL/SUBSYS/CELL/' + letter, '{:<15} {:<15} {:<15}'.format(*self.structure.cell[i]),
                            override=False, conflicting_keys=['ABC', 'ALPHA_BETA_GAMMA', 'CELL_FILE_NAME'])
            topo = "FORCE_EVAL/SUBSYS/TOPOLOGY"
            inp.add_keyword(topo + "/COORD_FILE_NAME", Cp2kCalculation._DEFAULT_COORDS_FILE_NAME, override=False)
            inp.add_keyword(topo + "/COORD_FILE_FORMAT", "XYZ", override=False, conflicting_keys=['COORDINATE'])
        return inp.render()


class InputSetsWithDefaultConfig(InputSetsFromFile):

    def __init__(self, structure, config, kind_section_config):
        """
        :param structure
        :param config
        :param kind_section_config: kind_section_config_path or TZV2PSets or DZVPSets
        """
        config_path = os.path.join(CONFIG_DIR, config)
        super(InputSetsWithDefaultConfig, self).__init__(structure, config_path=config_path,
                                                         kind_section_config=kind_section_config)


class EnergyInputSets(InputSetsWithDefaultConfig):
    def __init__(self, structure, config='energy.json', kind_section_config='DZVPSets'):
        super(EnergyInputSets, self).__init__(structure, config, kind_section_config)


class GeooptInputSets(InputSetsWithDefaultConfig):
    def __init__(self, structure_path, config='geoopt.json', kind_section_config='DZVPSets'):
        super(GeooptInputSets, self).__init__(structure_path, config, kind_section_config)


class NebInputSets(InputSetsWithDefaultConfig):
    def __init__(self, structure_path, config='neb.json', kind_section_config='DZVPSets'):
        super(NebInputSets, self).__init__(structure_path, config, kind_section_config)


class FrequencyInputSets(InputSetsWithDefaultConfig):
    def __init__(self, structure_path, config='frequency.json', kind_section_config='DZVPSets'):
        super(FrequencyInputSets, self).__init__(structure_path, config, kind_section_config)
