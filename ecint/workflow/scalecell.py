import os

import numpy as np
from aiida.engine import WorkChain
from aiida.orm import StructureData

from ecint.config import RESULT_NAME
from ecint.preprocessor.utils import check_config_machine, inspect_node
from ecint.workflow.units.base import EnergySingleWorkChain


class CelloptWorkChain(WorkChain):
    """
    Workflow to cell optimization
    """
    SUB = {'enercalc'}  # correspond to expose_inputs namespace

    @classmethod
    def define(cls, spec):
        super(CelloptWorkChain, cls).define(spec)

        spec.input('structure', valid_type=StructureData, required=True)

        # set single point energy calculation input setting
        # now only implement semiconductor k point sampling energy calculation
        spec.expose_inputs(EnergySingleWorkChain,
                           namespace='enercalc',
                           exclude=['structure', 'label'])

        spec.outline(
            cls.check_config_machine,
            cls.submit_cellopt,
            cls.inspect_cellopt,
            cls.write_results
        )

    def check_config_machine(self):
        check_config_machine(
            config=self.inputs.enercalc.config,
            machine=self.inputs.enercalc.machine
        )

    def submit_cellopt(self):
        """
        prepare the rough scaled structures and submit single point energy
        :return:
        """
        struct = self.inputs.structure.get_ase()

        # TODO: 0.95:1.05:0.01
        # initial the scaling parameter

        self.ctx.scale_list = np.arange(0.95, 1.05, 0.01)
        cell = struct.get_cell()
        for i, scale in enumerate(self.ctx.scale_list):
            tmp = struct.copy()
            tmp.set_cell(cell * scale, scale_atoms=True)
            tmp_struct_data = StructureData(ase=tmp)
            tmp_struct_data.store()
            node = self.submit(
                EnergySingleWorkChain,
                structure=tmp_struct_data,
                label=f'scale_{scale:.3f}',
                **self.exposed_inputs(EnergySingleWorkChain,
                                      namespace='enercalc')
            )
            self.to_context(**{f'cellopt_{i}': node})

    def inspect_cellopt(self):
        for i in range(len(self.ctx.scale_list)):
            inspect_node(self.ctx[f'cellopt_{i}'])

    def write_results(self):
        os.chdir(self.inputs.resdir)
        with open(RESULT_NAME, 'a') as f:
            f.write(f'# Scale Factor,  Volume, Energy')
            for i in range(len(self.ctx.scale_list)):
                volume = (
                    self.ctx[f'cellopt_{i}']
                        .inputs
                        .structure
                        .get_cell_volume()
                )
                energy = (
                    self.ctx[f'cellopt_{i}']
                        .outputs
                        .energy
                        .value
                )
                f.write(f'{i} {volume} {energy}')
