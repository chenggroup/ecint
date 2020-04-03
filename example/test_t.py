from aiida.common import AttributeDict
from aiida.engine import submit
from aiida_cp2k.workchains import Cp2kBaseWorkChain
from aiida.engine import while_
from aiida.plugins import CalculationFactory
from ecint.preprocessor import GeooptPreprocessor, NebPreprocessor, FrequencyPreprocessor
from ecint.preprocessor.input import GeooptInputSets, NebInputSets, FrequencyInputSets
from aiida_cp2k.workchains.aiida_base_restart import BaseRestartWorkChain

Cp2kCalculation = CalculationFactory('cp2k')


class NebWorkChain(Cp2kBaseWorkChain):
    _calculation_class = Cp2kCalculation

    @classmethod
    def define(cls, spec):
        super(NebWorkChain, cls).define(spec)
        spec.input('atoms')
        spec.input('machine')
        spec.outline(
            cls.run_geoopt(),
            while_(cls.run_geoopt().status=='FINISHED')(
                # cls.
            ),
            cls.run_neb
        )

    def run_geoopt(self):
        geoopt = GeooptPreprocessor(inputclass=GeooptInputSets(self.inputs.atoms), machine=self.inputs.machine)
        submit(geoopt.builder)

    def run_neb(self):
        neb = NebPreprocessor(inputclass=NebInputSets(self.inputs.atoms), machine=self.inputs.machine)
        submit(neb.builder)

    def run_frequency(self):
        frequency = FrequencyPreprocessor(inputclass=FrequencyInputSets(self.inputs.atoms), machine=self.inputs.machine)
        submit(frequency.builder)
