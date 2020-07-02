import os
from copy import deepcopy

from aiida_cp2k.utils import Cp2kInput

from ecint.preprocessor.kind import *
from ecint.preprocessor.kind import BaseSets
from ecint.preprocessor.utils import load_json, update_dict
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

    def __init__(self, structure, config_path, kind_section_config='DZVPBLYP'):
        """
        :param structure: atoms or StructureData
        :param config_path: base sets
        :param kind_section_config: kind_section sets, use path or TZV2PBLYP or DZVPBLYP DZVPPBE
        """
        self.structure = structure
        self.config = load_json(config_path)
        # TODO: add check for atoms.cell or atoms.get_volume
        # define self.kind_section
        if os.path.exists(kind_section_config):
            self.kind_section = SetsFromYaml(self.structure, kind_section_config).kind_section
        elif issubclass(eval(kind_section_config), BaseSets):
            __KindSectionSets = eval(kind_section_config)
            self.kind_section = __KindSectionSets(self.structure).kind_section
        else:
            raise ValueError('Unexpected kind_section_config, please input a yaml file or a builtin set')

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
            raise ValueError('FORCE_EVAL section should be dict or list')
        return _config

    def add_config(self, new_config):
        """
        :param new_config: like {'MOTION':{'BAND':{'NPROC_REP':24}}}
        """
        update_dict(self.config, new_config)

    def generate_cp2k_input_file(self):
        from aiida_cp2k.calculations import Cp2kCalculation
        inp = Cp2kInput(self.input_sets)
        for i, letter in enumerate('ABC'):
            # atoms or StructureData has the same property `cell`
            inp.add_keyword('FORCE_EVAL/SUBSYS/CELL/' + letter, '{:<15} {:<15} {:<15}'.format(*self.structure.cell[i]),
                            override=False, conflicting_keys=['ABC', 'ALPHA_BETA_GAMMA', 'CELL_FILE_NAME'])
            topo = "FORCE_EVAL/SUBSYS/TOPOLOGY"
            inp.add_keyword(topo + "/COORD_FILE_NAME", Cp2kCalculation._DEFAULT_COORDS_FILE_NAME, override=False)
            inp.add_keyword(topo + "/COORD_FILE_FORMAT", "XYZ", override=False, conflicting_keys=['COORDINATE'])
        return inp.render()


# maybe unneeded
# class InputSetsWithDefaultConfig(InputSetsFromFile):
#
#    def __init__(self, structure, config, kind_section_config):
#        """
#        :param structure
#        :param config
#        :param kind_section_config: kind_section_config_path or TZV2PBLYP or DZVPBLYP DZVPPBE
#        """
#        config_path = os.path.join(CONFIG_DIR, config)
#        super(InputSetsWithDefaultConfig, self).__init__(structure, config_path=config_path,
#                                                         kind_section_config=kind_section_config)


class EnergyInputSets(InputSetsFromFile):
    def __init__(self, structure, config=os.path.join(CONFIG_DIR, 'energy.json'), kind_section_config='DZVPBLYP'):
        super(EnergyInputSets, self).__init__(structure, config, kind_section_config)


class GeooptInputSets(InputSetsFromFile):
    def __init__(self, structure, config=os.path.join(CONFIG_DIR, 'geoopt.json'), kind_section_config='DZVPBLYP'):
        super(GeooptInputSets, self).__init__(structure, config, kind_section_config)


class NebInputSets(InputSetsFromFile):
    def __init__(self, structure, config=os.path.join(CONFIG_DIR, 'neb.json'), kind_section_config='DZVPBLYP'):
        super(NebInputSets, self).__init__(structure, config, kind_section_config)


class FrequencyInputSets(InputSetsFromFile):
    def __init__(self, structure, config=os.path.join(CONFIG_DIR, 'frequency.json'), kind_section_config='DZVPBLYP'):
        super(FrequencyInputSets, self).__init__(structure, config, kind_section_config)
