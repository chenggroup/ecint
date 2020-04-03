from ecint.preprocessor import GeooptPreprocessor, EnergyPreprocessor
from ecint.preprocessor.input import GeooptInputSets, EnergyInputSets
from ecint.preprocessor import LSFPreprocessor
from ase.io import read
from ase.build import molecule
from aiida.plugins import WorkflowFactory
from aiida.orm import (Code, Dict, SinglefileData, Str)
from aiida.engine import run, submit

from aiida import load_profile

load_profile()

# Cp2kBaseWorkChain = WorkflowFactory('cp2k.base')

input_files = {'structure_file': Str('../test/energy/h2o.xyz'),
               'config_file': Str('../test/energy/energy.json'),
               'kind_section_file': Str('../test/energy/kind_section.yaml'),
               'machine_file': Str('../test/energy/machine.json')}

atoms = read(input_files['structure_file'].value)
inputclass = EnergyInputSets(atoms, config='energy.json', kind_section_config=input_files['kind_section_file'].value)
pre = EnergyPreprocessor(inputclass)
pre.load_machine_from_json(input_files['machine_file'].value)
builder = pre.builder
submit(builder)

# inputclass = EnergyInputSets(atoms)
# builder = LSFPreprocessor(inputclass, machine).builder
# builder.settings = Dict(dict={'additional_retrieve_list': ["-pos-1.xyz"]})
# builder = EnergyPreprocessor(inputclass, machine).builder
# builder.cp2k.metadata.dry_run = True
# submit(builder)
