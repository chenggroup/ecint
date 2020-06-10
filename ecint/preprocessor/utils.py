from copy import deepcopy
import requests
import json
import yaml
from yaml import SafeLoader
from warnings import warn
from ase.io import read
from aiida.orm import Computer


def load_json(json_path):
    with open(json_path) as f:
        d = json.load(f)
    return d


def load_yaml(yaml_path):
    with open(yaml_path) as f:
        d = yaml.load(f, Loader=SafeLoader)
    return d


def update_dict(nested_dict, item):
    """
    update item to nested_dict
    :param nested_dict:
    :param item:
    :return:
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
    :param computer:
    :return: int
    """
    # TODO: LSF could not support default_procs_per_node, need change method
    computer = Computer.get(name=computer)
    default_procs_per_node = computer.get_default_mpiprocs_per_machine()
    if isinstance(default_procs_per_node, int):
        procs_per_node = default_procs_per_node
    elif default_procs_per_node is None:
        transport = computer.get_transport()
        with transport:
            retcode, stdout, stderr = transport.exec_command_wait("bhosts | awk '{print $4}' | sed -n '2p'")
            procs_per_node = int(stdout)
        computer.set_default_mpiprocs_per_machine(procs_per_node)
    return procs_per_node


def get_procs_per_node_from_code_name(code_computer):
    """
    :param code_computer: code@computer
    :return: node_core_num: int
    """
    code, computer = code_computer.split('@')
    # TODO: Need add more computer's procs_per_node, change it to builtin config
    if computer == 'chenglab51':
        procs_per_node = 24
    elif computer == 'chenglab52':
        procs_per_node = 28
    elif computer == 'aiida_test':
        procs_per_node = 28
    else:
        procs_per_node = get_procs_per_node(computer)
    return procs_per_node


def load_machine(machine_config):
    """
    :param machine_config: json file or yaml file or dict
    :return:
    """
    if isinstance(machine_config, dict):
        _machine = machine_config
    elif isinstance(machine_config, str):
        if machine_config.endswith('.yaml') or machine_config.endswith('.yml'):
            _machine = load_yaml(machine_config)
        elif machine_config.endswith('.json'):
            _machine = load_json(machine_config)
        else:
            raise ValueError('machine file should be .json or .yaml file')
    else:
        raise ValueError('Please use correct format of `machine`, dict or your .json/.yaml file path')
    restrict_machine = {}
    # set `code@computer`
    if 'code@computer' not in _machine:
        raise KeyError('You must set `code@computer`')
    else:
        restrict_machine.update({'code@computer': _machine['code@computer']})
    # set `nprocs`
    procs_per_node = get_procs_per_node_from_code_name(_machine['code@computer'])
    if 'nnode' in _machine:
        nprocs = _machine['nnode'] * procs_per_node
        if ('nprocs' in _machine) or ('n' in _machine):
            warn('You have set both `nnode` and `nprocs`(`n`), and the value of `nprocs`(`n`) will be ignored', Warning)
    elif ('nprocs' in _machine) or ('n' in _machine):
        nprocs = _machine.get('nprocs') or _machine.get('n')
    else:
        raise KeyError('You must set `nnode` or `nprocs` to appoint computing resources')
    restrict_machine.update({'tot_num_mpiprocs': nprocs})
    # set `walltime`
    if not (('walltime' in _machine) or ('max_wallclock_seconds' in _machine) or ('W' in _machine) or ('w' in _machine)):
        warn('You should set `walltime`, otherwise your job may waste computing resources', Warning)
    else:
        walltime = _machine.get('walltime') or _machine.get('max_wallclock_seconds') \
                   or _machine.get('W') or _machine.get('w')
        restrict_machine.update({'max_wallclock_seconds': walltime})
    # set `queue_name`
    if not (('queue' in _machine) or ('queue_name' in _machine) or ('q' in _machine)):
        warn('You have not set `queue`, so default value will be used', Warning)
    else:
        queue = _machine.get('queue') or _machine.get('queue_name') or _machine.get('q')
        restrict_machine.update({'queue_name': queue})
    # set `ptile`
    if 'ptile' in _machine:
        ptile = _machine.get('ptile')
    else:
        ptile = procs_per_node
    custom_scheduler_commands = f'#BSUB -R \"span[ptile={ptile}]\"'
    restrict_machine.update({'custom_scheduler_commands': custom_scheduler_commands})
    # return dict={'code@computer':, 'tot_num_mpiprocs': ,'max_wallclock_seconds': ,'queue_name': ,
    # 'custom_scheduler_commands': }
    return restrict_machine


def inp2json(cp2k_input):
    # TODO: need edit, parse cp2k input file to json format
    pass


def inspect_node(node):
    """
    inspect WorkChainNode is_finished_ok
    :param node: WorkChainNode
    :return:
    """
    # TODO: add some ExitCode
    assert node.is_finished_ok


def to_structure():
    # TODO: Atoms or str to StructureData
    pass


def check_machine():
    # TODO: check machine for a general situation
    pass


def check_neb(parameters, restrict_machine):
    """
    will update parameters['MOTION']['BAND']['NPROC_REP'] or restrict_machine['tot_num_mpiprocs']
    :param parameters:
    :param restrict_machine:
    :return:
    """
    # warning: if 'NPROC_REP' not set, than it will be setted as procs_per_node
    # set default nproc_rep as procs_per_node
    procs_per_node = get_procs_per_node_from_code_name(restrict_machine['code@computer'])
    if parameters['MOTION']['BAND'].get('NPROC_REP'):
        nproc_rep = parameters['MOTION']['BAND']['NPROC_REP']
    else:
        nproc_rep = parameters['MOTION']['BAND'].setdefault('NPROC_REP', procs_per_node)
        warn(f'Cause you have not set `/MOTION/BAND/NPROC_REP`, so it is setted as {nproc_rep}', ResourceWarning)
    # get number_of_replica and tot_num_mpiprocs
    number_of_replica = parameters['MOTION']['BAND']['NUMBER_OF_REPLICA']
    tot_num_mpiprocs = restrict_machine['tot_num_mpiprocs']
    # check whether nproc_rep*number_of_replica == 'tot_num_mpiprocs', if not, change tot_num_mpiprocs
    if nproc_rep * number_of_replica != tot_num_mpiprocs:
        restrict_machine['tot_num_mpiprocs'] = nproc_rep * number_of_replica
        warn(f'`/MOTION/BAND/NPROC_REP` ({nproc_rep}) * `/MOTION/BAND/NUMBER_OF_REPLICA` ({number_of_replica}) '
             f'does not correspond to `nnode` or `nprocs`, '
             f'so `tot_num_mpiprocs` is setted as {nproc_rep * number_of_replica}', ResourceWarning)


# def check_neb(parameters, machine):
# warning: if 'NPROC_REP' not set, than it will be setted as tot_num_mpiprocs / number_of_replica

# set default nnode/tot_num_mpiprocs
# number_of_replica = parameters['MOTION']['BAND']['NUMBER_OF_REPLICA']
# tot_num_mpiprocs = machine['tot_num_mpiprocs']
# auto generate nproc_rep
# nproc_rep = parameters['MOTION']['BAND'].setdefault('NPROC_REP', tot_num_mpiprocs / number_of_replica)
# if isinstance(nproc_rep, int):
#     raise ValueError(f'Number of process should be int')


def notification_in_dingtalk(webhook, node):
    headers = {'Content-Type': 'application/json'}
    title = 'Job Info'
    text = '## Job Info\n'
    text += 'Your job is over!\n'
    text += '>\n'
    text += f'> Job PK: **{node.pk}**\n'
    text += '>\n'
    try:
        node.called[0].inputs.cp2k__structure.get_formula()
        text += f'> Job Chemical Formula: **{node.called[0].inputs.cp2k__structure.get_formula()}**\n'
        text += '>\n'
    except AttributeError and IndexError:
        pass
    text += f'> Job Type: **{node.process_label}**\n'
    text += '>\n'
    text += f'> Job State: **{node.process_state.name}**\n'
    data = {'msgtype': 'markdown', 'markdown': {'title': title, 'text': text}}
    response = requests.post(url=webhook, headers=headers, data=json.dumps(data))
    return response
