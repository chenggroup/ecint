from aiida.engine import submit

from ecint.postprocessor import get_traj_for_energy_curve, get_max_energy_frame
from ecint.preprocessor import path2structure
from ecint.preprocessor.input import GeooptInputSets
from ecint.workflow import BaseWorkflow

structure = path2structure('')

newworkflow = BaseWorkflow(preprocessor=GeooptInputSets(structure), postprocessor=[
    get_traj_for_energy_curve, get_max_energy_frame]).builder

submit(newworkflow)
