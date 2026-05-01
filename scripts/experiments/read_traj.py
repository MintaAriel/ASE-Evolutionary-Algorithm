from ase.io.trajectory import Trajectory
from ase.visualize import view
from pathlib import Path
from ase.io import read
import os
import shutil
from ase.vibrations import Vibrations
from deepmd_template import DeepMDRelaxation, RelaxConfig
from ase.io import write

config = RelaxConfig(cores=[1,2,3])

deep = DeepMDRelaxation(config)
deep.model_key = 'deepmd_d3'
calc = deep.build_calculator(models_dir=Path('/home/vito/PythonProjects/ASEProject/container_gpu_2/models'),
                             device='cuda',
                             threads=2)

TRAJS_DIR = Path('/home/vito/PythonProjects/ASEProject/EA/test/nvidia parallel')

traj = '/home/vito/uspex_matlab/theo_pyxtal/2THP/test_2/output.traj'

traj = Trajectory(traj, 'r')

numeros = [0,8,26,29,54,65,74,85,92,125,130,132,136]

# for i in numeros:
#     atom18 = traj[i]
#     # view(atom18)
#     atom18_dir = f'/home/vito/PythonProjects/ASEProject/EA/data/theophylline/cif/test_2THP/str_{i}_POSCARS'
#     write(atom18_dir,atom18, format='vasp')


def phonons_at_gamma(atom, out_dir):

    vib_dir =  Path(out_dir) / 'vib'
    if vib_dir.exists() and vib_dir.is_dir():
        shutil.rmtree(vib_dir)
    vib = Vibrations(atom, name= vib_dir)
    vib.run()
    vib.summary()

    energy = vib.get_zero_point_energy()

    print('ZPE:', energy )
    return energy

# results_dir = '/home/vito/PythonProjects/ASEProject/EA/data/theophylline/cif/polymorphs'
#
# structures = os.listdir(results_dir)
results_dir = '/home/vito/PythonProjects/ASEProject/EA/data/theophylline/cif'
# structures =  ['128707.cif', '862238.cif', '1039798.cif']
# structures = ['/home/vito/PythonProjects/ASEProject/container_gpu_2/Results/128707/deepmd_d3_in/POSCAR_final',
#               '/home/vito/PythonProjects/ASEProject/container_gpu_2/Results/862238/deepmd_d3_in/POSCAR_final',
#               '/home/vito/PythonProjects/ASEProject/container_gpu_2/Results/1039798/deepmd_d3_in/POSCAR_final']
# print(structures)

from pathlib import Path

# Directory containing your files
folder = Path("/home/vito/PythonProjects/ASEProject/from_Carlo_to_Brian/Optd_9a_d")

# Lists to store paths
d_files = []
a9_files = []

# Loop over all .vasp files
for file in folder.glob("*.vasp"):
    name = file.name

    if name.endswith("_optd_d.vasp"):
        d_files.append(str(file))


    elif name.endswith("_optd_9a.vasp"):
        a9_files.append(str(file))
        print(file)

# Optional: sort for reproducibility
d_files.sort()
a9_files.sort()


print(d_files)

batch = []
ZPE = []
# for structure in a9_files:
#     # full_dir = os.path.join(results_dir, structure)
#     full_dir = structure
#     crystal = read(full_dir, format='vasp')
#     crystal.calc = calc
#     batch.append(crystal)
#     zpe = phonons_at_gamma(crystal, results_dir)
#     ZPE.append(zpe)
#     print(zpe)
#
# print(batch)
# print(ZPE)

