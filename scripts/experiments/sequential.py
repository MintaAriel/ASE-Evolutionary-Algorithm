


from ase.io import read
from ase.visualize import  view
from deepmd.calculator import DP
from ea.utils.config import load_config
from pathlib import Path
from ase.optimize import FIRE, LBFGS
from ase.filters import FrechetCellFilter
import warnings
warnings.filterwarnings("ignore", message=r"logm result may be inaccurate.*")

cfg = load_config()
# atom = new_batch[2]
# atoms_list = read('/home/vito/uspex_matlab/theo_pyxtal/2THP/test_1/all_relaxed.trajectory', index=':')


base_dir = Path(__file__).parent.parent.parent  # directory of your script
theophyline_dir = base_dir / cfg['paths']['data_dir'] / 'theophylline' / 'cif'

print(theophyline_dir)

structures = [f'str_{i}_POSCARS' for i in [6,10,18,19,26,29,38]]
calc = DP(model='/home/vito/PythonProjects/ASEProject/container_gpu_2/models/dpa3-d3_torch.pth', device='gpu')

results = []

for name in structures:
    str_path = theophyline_dir / name
    atom = read(str_path, format='vasp')

    n_molecules = len(atom) / 21
    atom.calc = calc
    energy = atom.get_potential_energy()
    energy_per_mol = energy / n_molecules

    results.append((name, n_molecules, energy, energy_per_mol))


# sort by total energy (ascending)
results.sort(key=lambda x: x[2])


# nicely formatted print
for name, n_molecules, energy, energy_per_mol in results:
    print(f"{name:<15}  Energy per {n_molecules:6.2f} mol: {energy:12.6f} eV   "
          f"Energy per molecule: {energy_per_mol:12.6f} eV")


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