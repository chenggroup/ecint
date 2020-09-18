import os
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from warnings import warn

from aiida_cp2k.utils import Cp2kInput

from ecint.preprocessor.kind import DZVPPBE, KindSection
from ecint.preprocessor.utils import load_config, update_dict
from ecint.workflow.units import CONFIG_DIR

__all__ = ['EnergyInputSets', 'GeooptInputSets', 'NebInputSets',
           'FrequencyInputSets']


class BaseInput(metaclass=ABCMeta):
    @property
    @abstractmethod
    def structure(self):
        """

        Returns:
            aiida.orm.StructureData: structure

        """
        pass

    @property
    @abstractmethod
    def config(self):
        """

        Returns:
            dict: base config dict

        """
        pass

    @property
    @abstractmethod
    def kind_section(self):
        """

        Returns:
            list[dict]:
                kind section list of dict,
                e.g. [{'_': 'H',
                       'BASIS_SET': 'TZV2P-MOLOPT-GTH',
                       'POTENTIAL': 'GTH-PBE'},
                      {'_': 'O',
                       'BASIS_SET': 'TZV2P-MOLOPT-GTH',
                       'POTENTIAL': 'GTH-PBE'}]

        """
        pass

    @property
    @abstractmethod
    def input_sets(self):
        """

        Returns:
            dict: combine structure, config and kind_section to get input_sets

        """
        pass

    @abstractmethod
    def generate_cp2k_input(self):
        """

        Returns:
            str: use input_sets to generate cp2k input

        """
        pass


class InputSets(BaseInput):
    # TODO: update dict with pbc in xyz, xy, yz, or zx
    def __init__(self, structure, config, kind_section):
        """

        Args:
            structure (aiida.orm.StructureData): input structure
            config (dict): input base config
            kind_section (KindSection or list): elements kind section

        """
        self._structure = structure
        self._config = config
        self._kind_section = kind_section
        self.check_global(self.config)

    @property
    def structure(self):
        return self._structure

    @property
    def config(self):
        return self._config

    def add_config(self, new_dict):
        """Update new_dict to self.config
        """
        update_dict(self.config, new_dict)
        return self.config

    @property
    def kind_section(self):
        if isinstance(self._kind_section, KindSection):
            if (self._kind_section.structure is not None) and (
                    self._kind_section.structure != self.structure):
                warn('You have set structure in KindSection, this structure '
                     'will be replaced by structure in InputSets',
                     UserWarning)
            self._kind_section.load_structure(self.structure)
            kind_section_list = self._kind_section.kind_section
        elif isinstance(self._kind_section, list):
            # evaluate if elements in _kind_section match elements in structure
            elements_in_kind_duplicate = [one_kind_section["_"] for
                                          one_kind_section in
                                          self._kind_section]
            elements_in_kind = set(elements_in_kind_duplicate)
            if len(elements_in_kind_duplicate) != len(elements_in_kind):
                raise ValueError('Duplicate elements in kind section')
            elements_in_structure = self.structure.get_symbols_set()
            if elements_in_kind == elements_in_structure:
                kind_section_list = self._kind_section
            elif elements_in_kind > elements_in_structure:
                kind_section_list = []
                for one_kind_section in self._kind_section:
                    if one_kind_section["_"] in elements_in_structure:
                        kind_section_list.append(one_kind_section)
            else:
                raise ValueError('Elements in kind section does not '
                                 'match elements in structure')
        else:
            raise TypeError('Input kind_section need be '
                            '`KindSection` or `dict`')
        return kind_section_list

    def _update_subsys(self, subsys):
        """Update subsys section
        """
        # add kind section info
        update_dict(subsys, {"KIND": self.kind_section})
        # add structure cell info
        for i, letter in enumerate('ABC'):
            update_dict(subsys, {"CELL": {letter: '{:<15} {:<15} {:<15}'
                        .format(*self.structure.cell[i])}})
        # TODO: add more pbc function, like treating with (False, False, True)
        if self.structure.pbc == (False, False, False):
            update_dict(subsys, {"CELL": {"PERIODIC": "NONE"}})
        # # add structure coordinate info
        # atoms = self.structure.get_ase()
        # tags = np.char.array([
        # '' if tag == 0 else str(tag) for tag in atoms.get_tags()])
        # symbols = np.char.array([
        # symbol.ljust(2) for symbol in atoms.get_chemical_symbols()]) + tags
        # positions = np.char.array(['{:20.10f} {:20.10f} {:20.10f}'
        #                            .format(p[0], p[1], p[2])
        #                            for p in atoms.get_positions()])
        # coords = symbols + positions
        # update_dict(subsys, {"COORD": {"": coords.tolist()}})

    @classmethod
    def check_global(cls, config):
        if config.get('GLOBAL'):
            raise ValueError('You can not set `GLOBAL` section '
                             'for a specific workflow')

    @property
    def input_sets(self):
        _input_sets = deepcopy(self.config)
        force_eval = _input_sets["FORCE_EVAL"]
        # update FORCE_EVAL or MULTI_FORCE_EVAL
        if isinstance(force_eval, dict):
            self._update_subsys(force_eval.setdefault("SUBSYS", {}))
        elif isinstance(force_eval, list):
            for one_force_eval in force_eval:
                self._update_subsys(one_force_eval.setdufault("SUBSYS", {}))
        else:
            raise TypeError('FORCE_EVAL section should be dict or list')
        return _input_sets

    def generate_cp2k_input(self):
        """Generate input_sets to cp2k input
        """
        inp = Cp2kInput(self.input_sets)
        return inp.render()


class UnitsInputSets(InputSets):
    """

    Args:
        structure (aiida.orm.StructureData): input structure
        config (str or dict): input base config file path
        kind_section (KindSection or list): elements kind section

    """
    TypeMap = {}

    @property
    def config(self):
        if isinstance(self._config, str):
            units_config_path = os.path.join(CONFIG_DIR,
                                             self.TypeMap[self._config])
            units_config = load_config(units_config_path)
        elif isinstance(self._config, dict):
            units_config = self._config
        else:
            raise TypeError(f'Units config should use config file under '
                            f'{CONFIG_DIR}')
        return units_config


class EnergyInputSets(UnitsInputSets):
    TypeMap = {'default': 'energy.json', 'metal': 'energy.json',
               'semiconductor': 'energy_smo_k.json'}

    def __init__(self, structure, config='metal', kind_section=DZVPPBE()):
        super(EnergyInputSets, self).__init__(structure, config, kind_section)
        self.add_config({"GLOBAL": {"RUN_TYPE": "ENERGY",
                                    "PRINT_LEVEL": "MEDIUM"}})


class GeooptInputSets(UnitsInputSets):
    TypeMap = {'default': 'geoopt.json'}

    def __init__(self, structure, config='test', kind_section=DZVPPBE()):
        super(GeooptInputSets, self).__init__(structure, config, kind_section)
        self.add_config({"GLOBAL": {"RUN_TYPE": "GEO_OPT",
                                    "PRINT_LEVEL": "MEDIUM"}})


class NebInputSets(UnitsInputSets):
    TypeMap = {'default': 'neb.json'}

    def __init__(self, structure, config='test', kind_section=DZVPPBE()):
        super(NebInputSets, self).__init__(structure, config, kind_section)
        self.add_config({"GLOBAL": {"RUN_TYPE": "BAND",
                                    "PRINT_LEVEL": "MEDIUM"}})


class FrequencyInputSets(UnitsInputSets):
    TypeMap = {'default': 'frequency.json'}

    def __init__(self, structure, config='test', kind_section=DZVPPBE()):
        super(FrequencyInputSets, self).__init__(structure, config,
                                                 kind_section)
        self.add_config({"GLOBAL": {"RUN_TYPE": "VIBRATIONAL_ANALYSIS",
                                    "PRINT_LEVEL": "MEDIUM"}})
