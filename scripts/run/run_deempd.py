from ea.backends.deepmd_client import DeepMDClient
from ea.utils.config import load_config
from ase.io import read

atom  = read ('/home/vito/PythonProjects/ASEProject/EA/results/THP/Grant/cell_params/stra9_transformed_bo_opt.cif')
# atom = read('/home/vito/PythonProjects/ASEProject/EA/results/THP/Grant/dft-poltarashka/experimental /862238.cif')
from deepmd.calculator import DP
from ase.io import write

calc = DP(model='/home/vito/PythonProjects/ASEProject/EA/models/dpa3-pbed3-pytorch.pth', device='gpu')
# calc = DP(model='/home/vito/PythonProjects/ASEProject/EA/models/dpa3-d3_torch.pth', device='gpu')

atom.calc = calc

energy = atom.get_potential_energy()
print(energy)
