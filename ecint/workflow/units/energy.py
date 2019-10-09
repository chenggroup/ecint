from aiida.engine import WorkChain


class SinglePointEnergy(WorkChain):

    @classmethod
    def define(cls, spec):
        super(SinglePointEnergy, cls).define(spec)
        spec.outline(
            cls.load_config,
            cls.generate_input_parameters,
            cls.submit_workchain,
            cls.inspect_workchain
        ),
