import os
import sys
from itertools import groupby
from os.path import exists, isabs, isdir
from pathlib import PurePath
from warnings import warn

import json5
import numpy as np
from aiida.orm import Computer, SinglefileData, StructureData
from ase import Atoms
from ase.io import read
from ase.io.extxyz import key_val_str_to_dict
from ruamel import yaml
from scipy.optimize import curve_fit

from ecint.config import default_cp2k_machine
from ecint.preprocessor.kind import KindSection


def load_json(json_path):
    with open(json_path) as f:
        # d = json.load(f, object_pairs_hook=OrderedDict)
        d = json5.load(f)
    return d


def load_yaml(yaml_path):
    with open(yaml_path) as f:
        # d = yaml.load(f, Loader=yaml.RoundTripLoader)
        d = yaml.load(f, Loader=yaml.SafeLoader)
    return d


def _preparse_xyz(filename, **kwargs):
    """
    Only xyz format supports elements like symbol + tag, for example, Fe2

    Args:
        filename: str or file-like object
        **kwargs:

    Returns:
        ase.Atoms:

    """
    try:
        atoms = read(filename, **kwargs)
    except KeyError:
        if isinstance(filename, PurePath):
            filename = str(filename)
        if filename == '-':
            filename = sys.stdin
        if isinstance(filename, str):
            with open(filename, 'r') as f:
                xyz_lines = f.readlines()
        else:
            filename.seek(0)
            xyz_lines = filename.readlines()
        kind_lines = np.array([line.strip().split() for line in xyz_lines[2:]])
        symbols, tags = [], []
        for kind_line in kind_lines[:, 0]:
            symbol_with_tag = [''.join(list(g)) for k, g in
                               groupby(kind_line, key=lambda x: x.isdigit())]
            symbols.append(symbol_with_tag[0])
            tags.append(int(symbol_with_tag[1])) \
                if len(symbol_with_tag) == 2 else tags.append(0)
        comment_line_info = key_val_str_to_dict(xyz_lines[1])
        atoms = Atoms(symbols=symbols, positions=kind_lines[:, 1:], tags=tags,
                      cell=comment_line_info.get("Lattice"),
                      pbc=comment_line_info.get("pbc"))
    return atoms


def load_structure(structure, cell=None, pbc=True, masses=None,
                   lazy_load=False, **kwargs):
    """

    Convert various structure formats to StructureData

    Args:
        structure (StructureData or Atoms or str): structure object
        cell (list) : cell parameters
        pbc (bool or list[bool]): pbc in x, y, z
        masses (dict): masses map for elements
        lazy_load (bool): if True, upload structure file to server directly,
            instead of load structure firstly

    Returns:
        StructureData or SinglefileData: structure data unstored

    """
    if lazy_load:
        return SinglefileData(file=os.path.abspath(structure))
    if isinstance(structure, StructureData):
        _structure = structure
    elif isinstance(structure, Atoms):
        _structure = StructureData(ase=structure)
    elif isinstance(structure, str):
        atoms = _preparse_xyz(structure, **kwargs)
        if masses:
            symbols = np.array(atoms.get_chemical_symbols())
            tags = atoms.get_tags()
            for element, mass in masses.items():
                symbol_with_tag = [''.join(list(g)) for k, g in
                                   groupby(element, key=lambda x: x.isdigit())]
                symbol = symbol_with_tag[0]
                tag = (int(symbol_with_tag[1])
                       if len(symbol_with_tag) == 2 else 0)
                s_index = np.argwhere(symbols == symbol)
                t_index = np.argwhere(tags == tag)
                for i in np.intersect1d(s_index, t_index):
                    atoms[i].mass = mass
        if not atoms.get_cell():
            atoms.set_cell(cell)
        atoms.set_pbc(pbc)
        _structure = StructureData(ase=atoms)
    else:
        raise TypeError('Please use correct format of `structure`, '
                        'ase.Atoms, aiida.orm.StructureData '
                        'or valid structure file')
    return _structure


def load_config(config):
    """

    Convert various config formats to dict

    Args:
        config (dict or str): dict, .json file, .yaml file

    Returns:
        dict: config

    """
    if isinstance(config, dict):
        config_dict = config
    elif isinstance(config, str):
        if config.endswith('.yaml') or config.endswith('.yml'):
            config_dict = load_yaml(config)
        elif config.endswith('.json'):
            config_dict = load_json(config)
        else:
            raise ValueError('Config file should be .json or .yaml file')
    else:
        raise TypeError('Please use correct format of `config`, '
                        'dict or your .json/.yaml file path')
    return config_dict


def load_kind(kind_section):
    """

    Convert various kind section formats to list[dict, ...]

    Args:
        kind_section (KindSection or list or dict or str): kind section object

    Returns:
        list[dict]: kind section list

    """
    if isinstance(kind_section, KindSection):
        kind_section_list = kind_section.kind_section
    elif isinstance(kind_section, list):
        kind_section_list = kind_section
    elif isinstance(kind_section, dict):
        kind_section_list = [{'_': element, **one_kind_section} for
                             element, one_kind_section in kind_section.items()]
    elif isinstance(kind_section, str):
        if kind_section.endswith('.yaml') or kind_section.endswith('.yml'):
            _kind_section = load_yaml(kind_section)
        elif kind_section.endswith('.json'):
            _kind_section = load_json(kind_section)
        else:
            raise ValueError('Kind section file should be .json or .yaml file')
        kind_section_list = load_kind(_kind_section)
    else:
        raise TypeError('Please use correct format of `kind section`, '
                        'list (e.g. [{"_": "H", '
                        '"BASIS_SET": "DZVP-MOLOPT-SR-GTH", '
                        '"POTENTIAL": "GTH-PBE-q1"}, ...]), '
                        'dict(e.g. {"H": {"BASIS_SET": "DZVP-MOLOPT-SR-GTH", '
                        '"POTENTIAL": "GTH-PBE-q1"}, ...}) '
                        'or your .json/.yaml file path')
    return kind_section_list


def is_valid_workdir(workdir):
    """

    Args:
        workdir (str): path name

    Returns:
        str: error information

    """
    if workdir is None:
        pass
    else:
        if not exists(workdir):
            return 'workdir is not exists'
        if not isdir(workdir):
            return 'workdir need be a directory'
        if not isabs(workdir):
            return 'workdir need be a absolute path'


def update_dict(nested_dict, item):
    """Update method for nested dict

    Update `item` to `nested_dict`, the value of `nested_dict` will be changed

    Args:
        nested_dict (dict): the dict which is waiting for change
        item (dict): the dict which will be update to `nested_dict`

    Returns:
        None

    """
    for key in item:
        value = item[key]
        sub_dict = nested_dict.get(key)
        if isinstance(sub_dict, dict):
            update_dict(sub_dict, value)
        elif isinstance(sub_dict, list) and isinstance(value, list):
            sub_dict.extend(value)
        elif not sub_dict:
            nested_dict.update(item)
        else:
            raise ValueError(f'Incoherent data {nested_dict} and {item}')


def get_procs_per_node(computer):
    """

    Get processes/node from computer name

    Args:
        computer (str): computer name

    Returns:
        int: processes per node

    """
    # TODO: LSF could not support default_procs_per_node, need change method
    computer = Computer.get(name=computer)
    default_procs_per_node = computer.get_default_mpiprocs_per_machine()
    if isinstance(default_procs_per_node, int):
        procs_per_node = default_procs_per_node
    elif default_procs_per_node is None:
        transport = computer.get_transport()
        with transport:
            retcode, stdout, stderr = transport.exec_command_wait(
                "bhosts | awk '{print $4}' | sed -n '2p'")
            procs_per_node = int(stdout)
        computer.set_default_mpiprocs_per_machine(procs_per_node)
    return procs_per_node


def get_procs_per_node_from_code_name(code_computer):
    """

    Get processes/node from code@computer

    Args:
        code_computer (str): `code@computer`

    Returns:
        int: processes per node

    """
    code, computer = code_computer.split('@')
    # TODO: Need add more computer's procs_per_node, change it to builtin config
    if computer == 'chenglab51':
        procs_per_node = 24
    elif computer == 'chenglab52':
        procs_per_node = 28
    elif computer == 'aiida_test':
        procs_per_node = 28
    elif computer == 'aiida_test_res':
        procs_per_node = 24
    else:
        procs_per_node = get_procs_per_node(computer)
    return procs_per_node


def load_machine(machine):
    """
    TODO: need simplify with pythonic way
    Convert user friendly machine to restrict machine

    Args:
        machine (dict or str): dict, .json file, .yaml file

    Returns:
        dict: restrict machine
        looks like,
            dict={
                'code@computer': ,
                'tot_num_mpiprocs': ,
                'max_wallclock_seconds': ,
                'queue_name': ,
                'custom_scheduler_commands':
            }

    """
    if isinstance(machine, dict):
        _machine = machine
    elif isinstance(machine, str):
        if machine.endswith('.yaml') or machine.endswith('.yml'):
            _machine = load_yaml(machine)
        elif machine.endswith('.json'):
            _machine = load_json(machine)
        else:
            raise ValueError('Machine file should be .json or .yaml file')
    else:
        raise TypeError('Please use correct format of `machine`, '
                        'dict or your .json/.yaml file path')
    restrict_machine = {}
    # set `code@computer`
    if 'code@computer' not in _machine:
        raise KeyError('You must set `code@computer`')
    else:
        restrict_machine.update({'code@computer': _machine['code@computer']})
    # set `nprocs`
    procs_per_node = \
        get_procs_per_node_from_code_name(_machine['code@computer'])
    if 'nnode' in _machine:
        nprocs = _machine['nnode'] * procs_per_node
        if ('nprocs' in _machine) or ('n' in _machine):
            warn('You have set both `nnode` and `nprocs`(`n`), '
                 'and the value of `nprocs`(`n`) will be ignored',
                 Warning)
    # TODO: make more specific for Slurm
    elif (('nprocs' in _machine) or ('n' in _machine) or
          ('tot_num_mpiprocs' in _machine)):
        nprocs = (_machine.get('nprocs') or _machine.get('n') or
                  _machine.get('tot_num_mpiprocs'))
    else:
        raise KeyError('You must set `nnode` or `nprocs` or `tot_num_mpiprocs` '
                       'to appoint computing resources')
    restrict_machine.update({'tot_num_mpiprocs': nprocs})
    # set `walltime`
    if not (('walltime' in _machine) or
            ('max_wallclock_seconds' in _machine) or
            ('W' in _machine) or ('w' in _machine)):
        warn('You should set `walltime`, '
             'otherwise your job may waste computing resources',
             Warning)
    else:
        walltime = (_machine.get('walltime') or
                    _machine.get('max_wallclock_seconds') or
                    _machine.get('W') or _machine.get('w'))
        restrict_machine.update({'max_wallclock_seconds': walltime})
    # set `queue_name`
    if not (('queue' in _machine) or ('queue_name' in _machine) or
            ('q' in _machine)):
        warn('You have not set `queue`, so default value will be used', Warning)
    else:
        queue = (_machine.get('queue') or _machine.get('queue_name')
                 or _machine.get('q'))
        restrict_machine.update({'queue_name': queue})
    # set custom_scheduler_commands
    # set `ptile`
    if 'ptile' in _machine:
        ptile = _machine.get('ptile')
        custom_scheduler_commands = f'#BSUB -R \"span[ptile={ptile}]\"'
    elif 'ngpu' in _machine:
        num_gpu = _machine.get('ngpu')
        custom_scheduler_commands = f'#SBATCH --gres=gpu:{num_gpu}'
    elif 'custom_scheduler_commands' in _machine:
        custom_scheduler_commands = _machine.get('custom_scheduler_commands')
    else:
        ptile = procs_per_node
        custom_scheduler_commands = f'#BSUB -R \"span[ptile={ptile}]\"'
    restrict_machine.update(
        {'custom_scheduler_commands': custom_scheduler_commands})
    # return dict={'code@computer': , 'tot_num_mpiprocs': ,
    # 'max_wallclock_seconds': ,'queue_name': ,'custom_scheduler_commands': }
    return restrict_machine


def inp2dict(cp2k_input):
    # TODO: need edit, parse cp2k input file to json format
    pass


def inspect_node(node):
    """inspect WorkChainNode is_finished_ok

    Use after running workchain, usually combine with tocontext

    Args:
        node (aiida.orm.ProcessNode): node returned by running workchain

    Returns:
        None

    """
    # TODO: add some ExitCode
    assert node.is_finished_ok


def check_config_machine(config=None, machine=None, uniform_func=None):
    """check config and machine

    If you set `uniform_func`, `config` and `restrict_machine` probably be changed

    Args:
        config (dict): input base config
        machine (dict): input machine
        uniform_func (Callable[[dict, dict], tuple]):
            uniform related paras in `parameters` and `restrict_machine`

    Returns:
        (dict, dict)

    Todo:
        check machine for a general situation,
        considerate all custom situation and use a general `uniform_func`

    """
    # get restrict_machine
    if machine is None:
        machine = default_cp2k_machine
        warn('Auto setup machine config', Warning)
    elif not isinstance(machine, dict):
        raise TypeError('Machine config need be dict')
    restrict_machine = load_machine(machine)
    # decide whether use uniform method or not
    if (config is not None) and (uniform_func is not None):
        uniform_func(config=config, restrict_machine=restrict_machine)
    print(f'Your machine config is: {restrict_machine}')
    return config, restrict_machine


def uniform_neb(config, restrict_machine):
    """Uniform resource related paras in `parameters` and `restrict_machine`

    Will update parameters['MOTION']['BAND']['NPROC_REP'] or
    restrict_machine['tot_num_mpiprocs']

    Args:
        config (dict): input parameters
        restrict_machine (dict): restrict machine,
            for example, dict after `load_machine`

    Returns:
        list[bool, bool]: True if changed, False otherwise;
            first for parameters and second for restrict_machine

    Todo:
        See also `Todo` in `check_machine`.
        If there is any similar general method,
        this method can be delete and replaced by it

    """
    # warning: if 'NPROC_REP' not set, than it will be setted as procs_per_node
    # set default nproc_rep as procs_per_node
    procs_per_node = \
        get_procs_per_node_from_code_name(restrict_machine['code@computer'])
    # first for parameters, second for restrict_machine
    is_changed = [False, False]
    if config['MOTION']['BAND'].get('NPROC_REP'):
        nproc_rep = config['MOTION']['BAND']['NPROC_REP']
    else:
        nproc_rep = config['MOTION']['BAND'].setdefault('NPROC_REP',
                                                        procs_per_node)
        is_changed[0] = True
        warn(f'Cause you have not set `/MOTION/BAND/NPROC_REP`, '
             f'so it is setted as {nproc_rep}',
             ResourceWarning)
    # get number_of_replica and tot_num_mpiprocs
    number_of_replica = config['MOTION']['BAND']['NUMBER_OF_REPLICA']
    tot_num_mpiprocs = restrict_machine['tot_num_mpiprocs']
    # check whether nproc_rep*number_of_replica == 'tot_num_mpiprocs',
    # if not, change tot_num_mpiprocs
    if nproc_rep * number_of_replica != tot_num_mpiprocs:
        restrict_machine['tot_num_mpiprocs'] = nproc_rep * number_of_replica
        is_changed[1] = True
        warn(f'`/MOTION/BAND/NPROC_REP` ({nproc_rep}) * '
             f'`/MOTION/BAND/NUMBER_OF_REPLICA` ({number_of_replica}) '
             f'does not correspond to `nnode` or `nprocs`, '
             f'so `tot_num_mpiprocs` is setted as '
             f'{nproc_rep * number_of_replica}',
             ResourceWarning)
    return is_changed


def birch_murnaghan_equation(V, V0, E0, B0, B0_prime):
    V_ratio = np.power(np.divide(V0, V), np.divide(2, 3))
    E = E0 + np.divide((9 * V0 * B0), 16) * (np.power(V_ratio - 1, 3) * B0_prime
                                             + np.power((V_ratio - 1), 2) * (
                                                     6 - 4 * V_ratio))
    return E


def fit_cell_bme(v_list, e_list):
    """
    :param v_list:
    :param e_list:
    :return: the fit parapmeter
    """
    popt, pcov = curve_fit(
        birch_murnaghan_equation,
        v_list,
        e_list,
        [v_list[0], e_list[0], 2, 2]
    )
    return popt, pcov
