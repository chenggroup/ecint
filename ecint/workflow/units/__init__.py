import os
from aiida_cp2k.workchains import Cp2kBaseWorkChain
from ecint.preprocessor import LSFPreprocessor
from aiida.orm import Dict
from aiida.engine import while_


CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))


class EnergyWorkChain(Cp2kBaseWorkChain):

    @classmethod
    def define(cls, spec):
        super(EnergyWorkChain, cls).define(spec)



