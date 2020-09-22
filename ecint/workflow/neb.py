from aiida.engine import ToContext, WorkChain
from aiida.orm import StructureData

from ecint.preprocessor.utils import check_config_machine, inspect_node, \
    uniform_neb
from ecint.workflow.units.base import FrequencySingleWorkChain, \
    GeooptSingleWorkChain, NebSingleWorkChain


class NebWorkChain(WorkChain):
    # SUB corresponds to expose_inputs namespace
    TYPE = 'simulation'
    SUB = {'geoopt', 'neb', 'frequency'}

    @classmethod
    def define(cls, spec):
        super(NebWorkChain, cls).define(spec)
        # use structures.image_0 as reactant,
        # image_1 as next point in energy curve, and so on
        # the last image_N as product
        spec.input_namespace('structures',
                             valid_type=StructureData,
                             dynamic=True)

        # set geoopt input files
        spec.expose_inputs(GeooptSingleWorkChain,
                           namespace='geoopt',
                           exclude=['structure', 'label'])
        # spec.input('geoopt.resdir',
        #            valid_type=str, required=True, non_db=True)
        # spec.input('geoopt.config',
        #            valid_type=dict, required=False, non_db=True)
        # spec.input('geoopt.kind_section', default=DZVPPBE(),
        #            valid_type=(list, KindSection),
        #            required=False, non_db=True)
        # spec.input('geoopt.machine', default=default_cp2k_machine,
        #            valid_type=dict, required=False, non_db=True)

        # set neb input files
        spec.expose_inputs(NebSingleWorkChain,
                           namespace='neb',
                           exclude=['structures', 'label'])
        # similar with geoopt section

        # set frequency input files
        spec.expose_inputs(FrequencySingleWorkChain,
                           namespace='frequency',
                           exclude=['structure', 'label'])
        # similar with geoopt section

        spec.outline(
            cls.check_config_machine,
            cls.submit_geoopt,
            cls.inspect_geoopt,
            cls.submit_neb,
            cls.inspect_neb,
            cls.submit_frequency,
            cls.inspect_frequency,
        )

        spec.expose_outputs(GeooptSingleWorkChain, namespace='reactant')
        spec.expose_outputs(GeooptSingleWorkChain, namespace='product')
        spec.expose_outputs(NebSingleWorkChain)
        spec.expose_outputs(FrequencySingleWorkChain)

    def check_config_machine(self):
        check_config_machine(config=self.inputs.neb.config,
                             machine=self.inputs.neb.machine,
                             uniform_func=uniform_neb)

    def submit_geoopt(self):
        reactant = self.inputs.structures['image_0']
        self.ctx.image_last_index = len(self.inputs.structures) - 1
        product = self.inputs.structures[f'image_{self.ctx.image_last_index}']
        node_reactant = self.submit(GeooptSingleWorkChain, structure=reactant,
                                    label='reactant',
                                    **self.exposed_inputs(GeooptSingleWorkChain,
                                                          namespace='geoopt'))
        node_product = self.submit(GeooptSingleWorkChain, structure=product,
                                   label='product',
                                   **self.exposed_inputs(GeooptSingleWorkChain,
                                                         namespace='geoopt'))
        return ToContext(geoopt_reactant_workchain=node_reactant,
                         geoopt_product_workchain=node_product)

    def inspect_geoopt(self):
        inspect_node(self.ctx.geoopt_reactant_workchain)
        self.out_many(
            self.exposed_outputs(self.ctx.geoopt_reactant_workchain,
                                 GeooptSingleWorkChain, namespace='reactant')
        )
        inspect_node(self.ctx.geoopt_product_workchain)
        self.out_many(
            self.exposed_outputs(self.ctx.geoopt_product_workchain,
                                 GeooptSingleWorkChain, namespace='product')
        )

    def submit_neb(self):
        structures_geoopt = {}
        reactant_geoopt = self.ctx.geoopt_reactant_workchain.outputs.structure_geoopt
        product_geoopt = self.ctx.geoopt_product_workchain.outputs.structure_geoopt
        structures_geoopt.update(
            {'image_0': reactant_geoopt,
             f'image_{self.ctx.image_last_index}': product_geoopt})
        for image_index in range(1, self.ctx.image_last_index):
            structures_geoopt.update(
                {f'image_{image_index}':
                     self.inputs.structures[f'image_{image_index}']})
        node = self.submit(NebSingleWorkChain, structures=structures_geoopt,
                           **self.exposed_inputs(NebSingleWorkChain,
                                                 namespace='neb'))
        self.to_context(neb_workchain=node)

    def inspect_neb(self):
        inspect_node(self.ctx.neb_workchain)
        self.out_many(
            self.exposed_outputs(self.ctx.neb_workchain, NebSingleWorkChain)
        )

    def submit_frequency(self):
        transition_state = self.ctx.neb_workchain.outputs.transition_state
        node = self.submit(FrequencySingleWorkChain, structure=transition_state,
                           **self.exposed_inputs(FrequencySingleWorkChain,
                                                 namespace='frequency'))
        self.to_context(frequency_workchain=node)

    def inspect_frequency(self):
        inspect_node(self.ctx.frequency_workchain)
        self.out_many(
            self.exposed_outputs(self.ctx.frequency_workchain,
                                 FrequencySingleWorkChain)
        )
