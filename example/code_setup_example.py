from aiida_lammps.tests.utils import (
    get_or_create_local_computer, get_or_create_code)
from aiida_lammps.tests.utils import lammps_version

lammps_path = "/share/apps/lammps-16Mar18/src/lmp_mpi"
work_directory = "/your/local/workdir"

computer = get_or_create_local_computer(work_directory, 'chenglab52')
code_lammps_force = get_or_create_code('lammps.force', computer, lammps_path)
code_lammps_opt = get_or_create_code('lammps.optimize', computer, lammps_path)
code_lammps_md = get_or_create_code('lammps.md', computer, lammps_path)

meta_options = {
    "resources": {
        "tot_num_mpiprocs": 56},
    "max_wallclock_seconds": 24*60*60,
    "queue_name": "small"
}
