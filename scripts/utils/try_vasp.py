from ea.io.vasp_run import create_vasp_sp, run_vasp_folders, read_runs
from ase.io import read, write


small_traj = read('/home/vito/uspex_matlab/theo_pyxtal/2THP/test_1 (Copy)/output.traj', index=':')[:5]

out_dir = '/home/vito/VASP/parallel_vasp_trial'

create_vasp_sp(out_dir, out_dir, small_traj)
run_vasp_folders(out_dir, small_traj, [0, 11], 2, 'b')
read_runs(small_traj, out_dir)