#from ase import read, write
from ase import Atoms


class ecint:
    def __init__(self, structure):
        if (isinstance(structure, Atoms)):
            self.structure = structure
        else:
            raise TypeError("This is not a Atoms object!")

