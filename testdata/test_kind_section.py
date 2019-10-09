from aiida.orm import StructureData
from ase.io import read

from ecint.preprocessor.kind import SetsFromYaml

structure_path = '~/aiidatest/h2o.xyz'
atoms = read(structure_path)
structure = StructureData(ase=atoms)

sets = SetsFromYaml(structure, None)
kind_section = sets.kind_section
print(kind_section)
