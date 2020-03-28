from aiida_cp2k.workchains import Cp2kBaseWorkChain
from aiida.engine import while_
from aiida.plugins import WorkflowFactory

# from aiida.plugins import WorkflowFactory

# Cp2kBaseWorkChain = WorkflowFactory('cp2k.base')
Cp2kBaseWorkChain = WorkflowFactory('cp2k.base')


class BaseWorkflow(object):
    def __init__(self, preprocessor, postprocessor):
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    @property
    def builder(self):
        return self.preprocessor.builder

    def get_results(self):
        for method in self.postprocessor:
            method()
