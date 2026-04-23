from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from huggingface_hub import login
from ase.visualize import view
from ase.io import read
import importlib.resources
from ase.ga.data import DataConnection
import os
from dotenv import load_dotenv
from ea.simulators.base import Simulator


class UmaRelax(Simulator):
    def __init__(self, model_name, device, key:str=None):
        self.model_name = model_name
        self.device = device
        if not key:
            self.key = self._load_api_key()

        self.predictor = pretrained_mlip.get_predict_unit(model_name=self.model_name, device=self.device)

    def _load_api_key(self):
        load_dotenv()
        key = os.getenv("UMA_API_KEY")
        if key is None:
            raise RuntimeError("UMA_API_KEY not set")
        return key

    def get_calc(self):
        self._load_api_key()
        calc = FAIRChemCalculator(self.predictor, task_name="omat")
        return calc






class RelaxDb():
    def __init__(self, db_dir):
        ...

