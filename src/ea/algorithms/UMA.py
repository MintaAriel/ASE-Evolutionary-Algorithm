from ase.build import bulk
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from huggingface_hub import login
from ase.visualize import view
from ase.io import read
import importlib.resources
from ase.ga.data import DataConnection
from fairchem.core import calculate
import json
from ase.io import write
from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit
from omegaconf import OmegaConf
# from ase.calculators.mixing import
da = DataConnection('/home/vito/PythonProjects/ASEProject/EA/MOLCRYS/Genesis/try/Nicotinamide_2/nicotinamide_1/nicotinamide_zahra.db')
# da = DataConnection('/home/vito/Downloads/Mg4Al8O16_401.db')
traj_file = '/home/vito/PythonProjects/ASEProject/CARLO/mytrajectory2.traj'
atom_unrel = read(traj_file, index='19')

atom_unrel = read('/home/vito/PythonProjects/ASEProject/CARLO/theophylline/CalcFold/best5000.cif', format='cif')
# atom_unrel = read('/home/vito/PythonProjects/ASEProject/CARLO/Carbamazepine/4_theof8.cif', format='cif')
# atom_unrel = da.get_atoms(3)
# view(atom_unrel)


predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda", cache_dir='/home/vito/Programs/UMA_models')

# print(predictor.overrides)\
'''
atom_ref_path = '/home/vito/Programs/UMA_models (Copy)/models--facebook--UMA/snapshots/38529caa2c51a9a8a0d71f0b56b79ac33bc9eceb/references/iso_atom_elem_refs.yaml'
ref = OmegaConf.load(atom_ref_path)
pointer_path = '/home/vito/.cache/fairchem/models--facebook--UMA/snapshots/38529caa2c51a9a8a0d71f0b56b79ac33bc9eceb/checkpoints/uma-s-1p1.pt'

predictor = MLIPPredictUnit(
            pointer_path,
            device='cuda',
            atom_refs=ref)


print(predictor)
'''

# HuggingFaceCheckpoint = pretrained_mlip.HuggingFaceCheckpoint
# PretrainedModels = pretrained_mlip.PretrainedModels
#
# with importlib.resources.files(calculate).joinpath("pretrained_models.json").open("rb") as f:
#     _MODEL_CKPTS = PretrainedModels(
#         checkpoints={
#             model_name: HuggingFaceCheckpoint(**hf_kwargs)
#             for model_name, hf_kwargs in json.load(f).items()
#         }
#     )
#
# print(_MODEL_CKPTS)
calc = FAIRChemCalculator(predictor, task_name="omat")

# #atoms = bulk("Fe")
# #
#atom_unrel = da.get_atoms(3)
#
# view(atoms)
# atoms.calc = calc
atom_unrel.calc = calc

opt = FIRE(FrechetCellFilter(atom_unrel), trajectory='/home/vito/PythonProjects/ASEProject/CARLO/4og_8mol.traj')
opt.run(0.005, 600)

# write('nicotinamide_c.vasp', atom_unrel)
