from aiida.engine import while_
from aiida.plugins import WorkflowFactory

Cp2kBaseWorkChain = WorkflowFactory('cp2k.base')


class NebWorkChain(Cp2kBaseWorkChain):

    @classmethod
    def define(cls, spec):
        super(Cp2kBaseWorkChain, cls).define(spec)
        spec.expose_inputs(Cp2kBaseWorkChain, namespace='cp2k')

        spec.outline(
            cls.setup,
            while_(cls.should_run_calculation)(
                cls.run_calculation,
                cls.inspect_calculation,
            ),
            cls.results,
        )

        spec.expose_outputs(Cp2kBaseWorkChain)
