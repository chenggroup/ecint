import os

import numpy as np
import matplotlib.pyplot as plt

from aiida.engine import WorkChain
from aiida.orm import StructureData

from ecint.preprocessor.utils import check_config_machine, inspect_node, \
    fit_cell_bme
from ecint.preprocessor.utils import birch_murnaghan_equation as bme
from ecint.workflow.units.base import EnergySingleWorkChain
from ecint.config import RESULT_NAME


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
            cls.submit_cellopt_1st,
            cls.inspect_cellopt_1st,
            cls.write_results_1st,
            cls.fit_cell_1st,
            cls.plot_cell_curve_1st,
            cls.submit_cellopt_2st,
            cls.inspect_cellopt_2st,
            cls.write_results_2st,
            cls.fit_cell_2st,
            cls.plot_cell_curve_2st
        )

    def check_config_machine(self):
        check_config_machine(
            config=self.inputs.enercalc.config,
            machine=self.inputs.enercalc.machine
        )

    def submit_cellopt_1st(self):
        """
        prepare the rough scaled structures and submit single point energy
        :return:
        """
        struct = self.inputs.structure.get_ase()

        # initial the scaling parameter
        self.ctx.scale_list_1st = np.arange(0.95, 1.05, 0.02)
        self.ctx.scale_list = self.ctx.scale_list_1st
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
            self.to_context(**{f'cellopt_{scale:.3f}': node})

    def inspect_cellopt_1st(self):
        for scale in self.ctx.scale_list:
            inspect_node(self.ctx[f'cellopt_{scale:.3f}'])

    def write_results_1st(self):
        os.chdir(self.inputs.enercalc.resdir)
        with open(RESULT_NAME, 'a') as f:
            f.write(f'# First Iteration of Cell Optimization\n')
            f.write(f'# Scale Factor,  Volume[Angstrom^3],  Energy[eV]\n')
            volume_list = []
            energy_list = []
            for i, scale in enumerate(self.ctx.scale_list):
                volume = (
                        self.ctx[f'cellopt_{scale:.3f}']
                        .inputs
                        .structure
                        .get_cell_volume()
                )
                volume_list.append(volume)
                energy = (
                    self.ctx[f'cellopt_{scale:.3f}']
                        .outputs
                        .energy
                        .value
                )
                energy_list.append(energy)

                f.write(f'{scale} {volume} {energy}\n')
            self.ctx.volume_list = volume_list
            self.ctx.energy_list = energy_list

    def fit_cell_1st(self):
        """
        fit the first rough volume from 0.95 to 1.05
        :return:
        """
        self.ctx.popt, self.ctx.pcov = fit_cell_bme(
            self.ctx.volume_list,
            self.ctx.energy_list
        )

        os.chdir(self.inputs.enercalc.resdir)
        with open(RESULT_NAME, 'a') as f:
            f.write(f'# The best fit volume is {self.ctx.popt[0]}\n')

    def plot_cell_curve_1st(self):
        plt.figure()
        plt.scatter(
            self.ctx.volume_list,
            self.ctx.energy_list,
            label="DFT"
        )
        plt.plot(
            self.ctx.volume_list,
            bme(
                self.ctx.volume_list,
                *self.ctx.popt
            ),
            label="B-M Fitting"
        )
        # font setting
        plt.title('System', fontsize=15)
        plt.ylabel("Eneryg [eV]", fontsize=15)
        plt.xlabel(r"Volume [$\AA ^3$]", fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend()
        plt.savefig(
            os.path.join(
                self.inputs.enercalc.resdir,
                "cell_fit_1st.png"
            ),
            format='png',
            dpi=72
        )

    def submit_cellopt_2st(self):

        struct = self.inputs.structure.get_ase()
        v = struct.get_volume()
        best_v_1st = self.ctx.popt[0]
        best_scale_1st = np.cbrt(best_v_1st/v)
        self.ctx.scale_list_2st = np.arange(
            best_scale_1st-0.01,
            best_scale_1st+0.01,
            0.002
        )
        self.ctx.scale_list = self.ctx.scale_list_2st


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
            self.to_context(**{f'cellopt_{scale:.3f}': node})

    def inspect_cellopt_2st(self):
        for scale in self.ctx.scale_list:
            inspect_node(self.ctx[f'cellopt_{scale:.3f}'])

    def write_results_2st(self):
        os.chdir(self.inputs.enercalc.resdir)
        with open(RESULT_NAME, 'a') as f:
            f.write(f'# Second Iteration of Cell Optimization\n')
            f.write(f'# Scale Factor,  Volume[Angstrom^3],  Energy[eV]\n')
            volume_list = []
            energy_list = []
            for i, scale in enumerate(self.ctx.scale_list):
                volume = (
                        self.ctx[f'cellopt_{scale:.3f}']
                        .inputs
                        .structure
                        .get_cell_volume()
                )
                volume_list.append(volume)
                energy = (
                    self.ctx[f'cellopt_{scale:.3f}']
                        .outputs
                        .energy
                        .value
                )
                energy_list.append(energy)

                f.write(f'{scale} {volume} {energy}\n')
            # add new volume/energy_list to old one
            self.ctx.volume_list += volume_list
            self.ctx.energy_list += energy_list

    def fit_cell_2st(self):
        """
        fit the second volume from best-0.01 to best+0.01
        :return:
        """
        self.ctx.popt, self.ctx.pcov = fit_cell_bme(
            self.ctx.volume_list,
            self.ctx.energy_list
        )

        os.chdir(self.inputs.enercalc.resdir)
        with open(RESULT_NAME, 'a') as f:
            f.write(f'# The best fit volume is {self.ctx.popt[0]}\n')

    def plot_cell_curve_2st(self):

        # sort the argument with increase order
        v_list = np.array(self.ctx.volume_list)
        e_list = np.array(self.ctx.energy_list)
        sort_idx = v_list.argsort()
        v_list = v_list[sort_idx]
        e_list = e_list[sort_idx]

        plt.figure()
        plt.scatter(
            v_list,
            e_list,
            label="DFT"
        )
        plt.plot(
            v_list,
            bme(
                v_list,
                *self.ctx.popt
            ),
            label="B-M Fitting"
        )
        # font setting
        plt.title('System', fontsize=15)
        plt.ylabel("Eneryg [eV]", fontsize=15)
        plt.xlabel(r"Volume [$\AA ^3$]", fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend()
        plt.savefig(
            os.path.join(
                self.inputs.enercalc.resdir,
                "cell_fit_2st.png"
            ),
            format='png',
            dpi=72
        )
