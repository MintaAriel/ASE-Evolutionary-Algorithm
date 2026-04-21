from ase.io.trajectory import Trajectory
from ase.visualize import view
from pathlib import Path
from ase.io import write

TRAJS_DIR = Path('/home/vito/PythonProjects/ASEProject/EA/test/nvidia parallel')

traj = TRAJS_DIR / '41'/ 'output.traj'

traj = Trajectory(traj, 'r')

atom18 = traj[10]
atom18_dir = '/home/vito/PythonProjects/ASEProject/EA/data/theophylline/cif/str_10_POSCARS'
write(atom18_dir,atom18, format='vasp')

