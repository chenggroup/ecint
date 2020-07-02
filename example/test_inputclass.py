from ase.build import molecule

from ecint.preprocessor.input import FrequencyInputSets

atoms = molecule('H2O')
atoms.center(vacuum=2.0)

inputclass = FrequencyInputSets(atoms)
print(inputclass.generate_cp2k_input_file())
