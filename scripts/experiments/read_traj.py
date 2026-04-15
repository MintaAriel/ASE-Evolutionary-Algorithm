from ase.io.trajectory import Trajectory
from ase.visualize import view
from pathlib import Path

TRAJS_DIR = Path('/home/vito/Documents/Shorts/traj/Teohpylline depmd d3')

traj = TRAJS_DIR / 'opt_10.traj'

traj = Trajectory(traj, 'r')

view(traj)


