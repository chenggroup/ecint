from ecint.preprocessor.utils import _preparse_xyz

with open('resources/Hematite.xyz', 'r') as f:
    atoms = _preparse_xyz(f, format='xyz')
