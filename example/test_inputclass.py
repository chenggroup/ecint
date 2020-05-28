from ecint.preprocessor.input import EnergyInputSets, GeooptInputSets, NebInputSets, FrequencyInputSets
from ase.build import molecule

atoms = molecule('H2O')
atoms.center(vacuum=2.0)

inputclass = FrequencyInputSets(atoms)
print(inputclass.generate_cp2k_input_file())