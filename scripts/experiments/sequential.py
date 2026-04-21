


from ase.io import read
from ase.visualize import  view
from deepmd.calculator import DP
from pathlib import Path
from ase.optimize import FIRE, LBFGS
from ase.filters import FrechetCellFilter
import warnings
warnings.filterwarnings("ignore", message=r"logm result may be inaccurate.*")



uspex_test = '/home/vito/uspex_matlab/theo_uspex/test_parallel'

new_batch = [read(Path(uspex_test, f'CalcFold{i}', 'geom.in'), format='vasp') for i in range(1,39)]

calc = DP(model='/home/vito/PythonProjects/ASEProject/container_gpu_2/models/dpa3-d3_torch.pth', device='gpu')

# atom = new_batch[2]
atoms_list = read('/home/vito/uspex_matlab/theo_pyxtal/2THP/test_1/all_relaxed.trajectory', index=':')
# atom = read('/home/vito/PythonProjects/ASEProject/EA/data/theophylline/cif/str_19_POSCARS', format='vasp')
atom = atoms_list[17]
view(atom)
atom.calc = calc
print(atom.get_potential_energy())

# fire = FIRE(
#     FrechetCellFilter(atom),
#     maxstep=0.03,
# )
# fire.run(fmax=0.10, steps=500)
# lbfgs = LBFGS(
#             FrechetCellFilter(atom),
#             append_trajectory=True,
#             maxstep=0.03,
#             memory=40,
#         )
#
# lbfgs.run(fmax=0.01, steps=200)
# view(atom)