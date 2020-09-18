import json
import re
from dataclasses import dataclass
from warnings import warn

import click
from ruamel import yaml


@dataclass(frozen=True)
class InsertValue:
    key: str
    value: str or dict


def flatten_seq(sequence):
    for item in sequence:
        if type(item) is list:
            for subitem in flatten_seq(item):
                yield subitem
        else:
            yield item


class Cp2kInp(object):
    # TODO: need test
    def __init__(self, filename):
        self.filename = filename
        self.set_var_val = {}
        self.if_flag, self.end_flag = True, True

    def __repr__(self):
        return ''.join(self.lines)

    @property
    def lines(self):
        with open(self.filename) as f:
            lines = f.readlines()
        return lines

    @property
    def well_defined_lines(self):
        return self._get_well_defined_lines(self.lines)

    def get_config(self):
        tree = self.get_tree()
        force_eval = tree["FORCE_EVAL"]
        try:
            if isinstance(force_eval, dict):
                force_eval.pop("SUBSYS")
            elif isinstance(force_eval, list):
                for one_force_eval in force_eval:
                    one_force_eval.pop("SUBSYS")
        except KeyError:
            pass
        return tree

    def extract_kind_section(self):
        force_eval = self.get_tree()["FORCE_EVAL"]
        if isinstance(force_eval, list):
            force_eval = force_eval[0]
        try:
            kind_section_list = force_eval["SUBSYS"]["KIND"]
            kind_section_dict = {kind_section.pop('_'): kind_section for
                                 kind_section in kind_section_list}
            return kind_section_dict
        except KeyError:
            warn('No &KIND info found, so kind section will not be parsed',
                 Warning)

    def get_tree(self):
        return self.get_tree_from_lines(self.well_defined_lines)

    @classmethod
    def get_tree_from_lines(cls, well_defined_lines):
        tree = {}
        for line in well_defined_lines:
            if line.upper().startswith("&END"):
                break
            elif line.upper().startswith("&"):
                name = line.split(None, 1)[0][1:].upper()
                cls._parse_section_start(line, tree)
                if isinstance(tree[name], dict):
                    tree[name].update(
                        cls.get_tree_from_lines(well_defined_lines))
                elif isinstance(tree[name], list):
                    tree[name][-1].update(
                        cls.get_tree_from_lines(well_defined_lines))
            else:
                cls._parse_keyword(line, tree)
        return tree

    @classmethod
    def _parse_section_start(cls, section_start, tree):
        section_pair = section_start.split(None, 1)
        name = section_pair[0][1:].upper()
        section_init = {'_': section_pair[1]} if len(section_pair) == 2 else {}
        tree = cls._update_tree(tree, InsertValue(name, section_init))
        return tree

    @classmethod
    def _parse_keyword(cls, keyword, tree):
        keyword_pair = keyword.split(None, 1)
        name = keyword_pair[0].upper()
        value = keyword_pair[1] if len(keyword_pair) == 2 else ''
        tree = cls._update_tree(tree, InsertValue(name, value))
        return tree

    @classmethod
    def _update_tree(cls, tree, insertval):
        name = insertval.key
        value = insertval.value
        if tree.get(name) and isinstance(tree[name], type(value)):
            tree[name] = [tree[name], value]
        elif tree.get(name) and isinstance(tree[name], list):
            tree[name].append(value)
        else:
            tree[name] = value
        return tree

    def _get_well_defined_lines(self, lines):
        # parse single line
        lines = self._flatten_lines(lines)
        # clean blank lines
        well_defined_lines = filter(None, lines)
        return well_defined_lines

    def _flatten_lines(self, lines):
        lines = list(flatten_seq(lines))
        lines = list(map(self._parse_line, lines))
        if any(type(line) is list for line in lines):
            return self._flatten_lines(lines)
        return lines

    def _parse_line(self, line):
        line = self._remove_comment(line)
        # convert SET
        if line.upper().startswith("@SET"):
            self._convert_set(line)
            line = ''
        else:
            line = self._convert_var(line)
        # convert IF
        if line.upper().startswith("@IF"):
            if not self.end_flag:
                raise ValueError("Do not use nested @IF")
            self.if_flag, self.end_flag = self._convert_if(line), False
            line = ''
        elif line.upper().startswith("@ENDIF"):
            if self.end_flag:
                raise ValueError("Can not find @IF before @ENDIF")
            self.if_flag, self.end_flag = True, True
            line = ''
        if not self.if_flag:
            line = ''
        # convert INCLUDE
        if line.upper().startswith("@INCLUDE"):
            line = self._convert_include(line)
        return line

    @classmethod
    def _remove_comment(cls, line):
        return line.split('!', 1)[0].split('#', 1)[0].strip()

    def _convert_set(self, line):
        variable, value = line.split(None, 2)[1:]
        self.set_var_val.update({variable: value})

    def _convert_var(self, line):
        user_var = re.search(r'\$(\{)?(?P<name>\w+)(?(1)\}|)', line)
        if user_var and (user_var['name'] not in self.set_var_val):
            raise ValueError(f'Variable {user_var} used before defined')
        for variable, value in self.set_var_val.items():
            line = re.sub(r'\$(\{)?(%s)(?(1)\}|)' % variable, value, line)
        return line

    @classmethod
    def _convert_if(cls, line):
        if len(line.split(None, 1)) == 1:
            if_express = False
        else:
            if_express = False if line.split(None, 1)[1] == '0' else True
        return if_express

    @classmethod
    def _convert_include(cls, line):
        filename = line.split(None, 1)[1]
        try:
            with open(filename, 'r') as f:
                file_lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f'No @INCLUDE File: {filename}')
        return [cls._remove_comment(line) for line in file_lines]


@click.command()
@click.argument('filename', type=click.Path(exists=True))
@click.argument('config_name', default='ecint.json', type=click.Path())
@click.option('--format', '-f', 'fm', help='output config format')
@click.option('--kind', '-k', 'kind_section', type=click.Path(),
              help='output kind section name')
def inp2config(filename, config_name, fm, kind_section):
    cp2k_inp = Cp2kInp(filename)
    yml = yaml.YAML()
    yml.indent(mapping=2, sequence=4, offset=2)
    config_tree = cp2k_inp.get_config()
    with open(config_name, 'w') as f:
        if fm == 'json' or config_name.endswith('.json'):
            json.dump(config_tree, f, indent=2)
        elif fm == 'yaml' or config_name.endswith('.yaml'):
            yml.dump(config_tree, f)
        else:
            raise ValueError('Unknown config file type, '
                             'please use `.json` or `.yaml`')
    kind_tree = cp2k_inp.extract_kind_section()
    if kind_section and kind_tree:
        with open(kind_section, 'w') as f:
            if kind_section.endswith('.json'):
                json.dump(kind_tree, f, indent=2)
            elif kind_section.endswith('.yaml'):
                yml.dump(kind_tree, f)
            else:
                raise ValueError('Unknown kind section file type, '
                                 'please use `.json` or `.yaml`')
