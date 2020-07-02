from aiida import load_profile
from ase.build import molecule

from ecint.preprocessor.input import NebInputSets
from ecint.workflow.units import JsonDryRun

load_profile()
atoms = molecule('H2O')
atoms.center(vacuum=5)

nis = NebInputSets(atoms)
jdr = JsonDryRun(nis)

jdr.run()
