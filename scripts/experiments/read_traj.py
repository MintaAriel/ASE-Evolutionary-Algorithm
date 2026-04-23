from ase.io.trajectory import Trajectory
from ase.visualize import view
from pathlib import Path
from ase.io import write

TRAJS_DIR = Path('/home/vito/PythonProjects/ASEProject/EA/test/nvidia parallel')

traj = '/home/vito/uspex_matlab/theo_pyxtal/2THP/test_2/output.traj'

traj = Trajectory(traj, 'r')

numeros = [0,8,26,29,54,65,74,85,92,125,130,132,136]

for i in numeros:
    atom18 = traj[i]
    # view(atom18)
    atom18_dir = f'/home/vito/PythonProjects/ASEProject/EA/data/theophylline/cif/test_2THP/str_{i}_POSCARS'
    write(atom18_dir,atom18, format='vasp')

