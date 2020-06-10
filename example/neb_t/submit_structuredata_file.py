from os.path import join
from aiida.orm import StructureData, Dict, Code
from aiida.engine import run, submit
from ecint.workflow.units import CONFIG_DIR
from ecint.preprocessor.utils import load_json, update_dict
from ase.io import read
from aiida import load_profile

load_profile()

reactant_atoms = read('is.xyz')
product_atoms = read('fs.xyz')
guessedts_atoms = read('Replica3.xyz')
reactant = StructureData(ase=reactant_atoms)
product = StructureData(ase=product_atoms)
guessedts = StructureData(ase=guessedts_atoms)

config = load_json(join(CONFIG_DIR, 'neb.json'))
update_dict(config, {
    'FORCE_EVAL': {'SUBSYS': {'KIND': [{'_': 'H', 'BASIS_SET': 'DZVP-MOLOPT-SR-GTH', 'POTENTIAL': 'GTH-PBE-q1'},
                                       {'_': 'O', 'BASIS_SET': 'DZVP-MOLOPT-SR-GTH', 'POTENTIAL': 'GTH-PBE-q6'},
                                       {'_': 'Ti', 'BASIS_SET': 'DZVP-MOLOPT-SR-GTH', 'POTENTIAL': 'GTH-PBE-q12'}]}}})
update_dict(config, {'MOTION': {'BAND': {'NPROC_REP': 28}}})
update_dict(config, {'MOTION': {'BAND': {'REPLICA': [{'COORD_FILE_NAME': 'reactant.xyz'},
                                                     {'COORD_FILE_NAME': 'product.xyz'},
                                                     {'COORD_FILE_NAME': 'guessedts.xyz'}]}}})

parameters = Dict(dict=config)
code = Code.get_from_string('cp2k@aiida_test')

builder = code.get_builder()
builder.structure = reactant
builder.parameters = parameters
builder.code = code
builder.metadata.options.resources = {
    'tot_num_mpiprocs': 28 * 6,
    "num_mpiprocs_per_machine": 28,
}
builder.metadata.options.queue_name = 'large'
builder.metadata.options.max_wallclock_seconds = 24 * 60 * 60
builder.metadata.options.custom_scheduler_commands = f'#BSUB -R "span[ptile=28]"'
builder.file = {'reactant': reactant, 'product': product, 'guessedts': guessedts}

# builder.metadata.dry_run = True
# run(builder)
submit(builder)
