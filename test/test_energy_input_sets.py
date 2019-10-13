from ecint.preprocessor.input import EnergyInputSets
from ecint.preprocessor import path2structure

structure = path2structure('./data/h2o.xyz')

eis = EnergyInputSets(structure)
input_sets = eis.generate_cp2k_input_file()
print(input_sets)
