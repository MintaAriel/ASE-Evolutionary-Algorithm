from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from huggingface_hub import login
from ase.visualize import view
from ase.io import read
import importlib.resources
from ase.ga.data import DataConnection

class UmaRelax():
    def __init__(self, model_name, device, key):
        self.model_name = model_name
        self.device = device
        self.key = key
        self.predictor = pretrained_mlip.get_predict_unit(model_name=self.model_name, device=self.device)

    def get_calc(self):
        calc = FAIRChemCalculator(self.predictor, task_name="omat")
        return calc






class RelaxDb():
    def __init__(self, db_dir):
        ...

