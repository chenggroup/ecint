RESULT_NAME = 'results.dat'
default_cp2k_machine = {'code@computer': 'cp2k@aiida_test', 'nnode': 1,
                        'walltime': 12 * 60 * 60, 'queue': 'medium'}
default_cp2k_large_machine = {'code@computer': 'cp2k@aiida_test', 'nnode': 4,
                              'queue': 'large'}

default_dpmd_gpu_machine = {
    'code@computer': 'deepmd@metal',
    'nprocs': 2,
    'queue': 'test',
    'ngpu': 1
}
