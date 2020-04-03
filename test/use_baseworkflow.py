from ecint.workflow import BaseWorkflow
from ecint.preprocessor.input import GeooptInputSets
from ecint.preprocessor import path2structure
from ecint.postprocessor import get_traj_for_energy_curve, get_max_energy_frame
from aiida.engine import submit

structure = path2structure('')

newworkflow = BaseWorkflow(preprocessor=GeooptInputSets(structure), postprocessor=[
                           get_traj_for_energy_curve, get_max_energy_frame]).builder

submit(newworkflow)
