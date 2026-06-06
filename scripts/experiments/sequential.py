

from ea.io.uspex_io import get_structure_from_id
from ase.io import read
from ase.visualize import  view
from deepmd.calculator import DP
from ea.utils.config import load_config
from pathlib import Path
from io import StringIO
from ea.parallel.create_batch import batch_calculator_deepmd_threaded
from ase.optimize import FIRE, LBFGS
from ase.filters import FrechetCellFilter
import warnings
import time
warnings.filterwarnings("ignore", message=r"logm result may be inaccurate.*")
#calc = DP(model='/home/vito/PythonProjects/ASEProject/container_gpu_2/models/dpa3-d3_torch.pth', device='gpu')

cfg = load_config()
# atom = new_batch[2]
# atoms_list = read('/home/vito/uspex_matlab/theo_pyxtal/2THP/test_1/all_relaxed.trajectory', index=':')


base_dir = Path(__file__).parent.parent.parent  # directory of your script
theophyline_dir = base_dir / cfg['paths']['data_dir'] / 'theophylline' / 'cif'

print(theophyline_dir)


def convert_poly():
    structures = [f'str_{i}_POSCARS' for i in [6,10,18,19,26,29,38]]

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



def check_uspex(parallel=True):
    data = get_structure_from_id(poscar_dir='/home/vito/PythonProjects/ASEProject/EA/test/THP2-sirena/gatheredPOSCARS',
                                 id_structures=[i for i in range(600,1000,1)])
    batch = []
    for k,v in data.items():
        atom = read(StringIO(v), format='vasp')
        batch.append(atom)

    print(batch)

    if parallel:
        start = time.perf_counter()

        E, F, V = batch_calculator_deepmd_threaded(batch, calc, n_workers=3)

        print(E)
        end = time.perf_counter()
        print(f"Elapsed time: {end - start} seconds")

    else:
        energies = []
        start = time.perf_counter()
        for atom in batch:
            atom.calc = calc
            energy = atom.get_potential_energy()
            print(energy)
            energies.append(energy)

        print(energies)
        end = time.perf_counter()
        print(f"Elapsed time: {end - start} seconds")




if __name__ == '__main__':
    from deepmd_parallel import run, RelaxConfig
    start2 = time.perf_counter()
    # calc = DP(model='/home/vito/PythonProjects/ASEProject/EA/models/dpa3-pbed3-pytorch.pth', device='gpu')
    # # calc = DP(model='/home/vito/PythonProjects/ASEProject/EA/models/dpa3-d3_torch.pth', device='gpu')
    # # nuevo = read('/home/vito/Downloads/1842250.cif')
    # nuevo = read('/home/vito/PythonProjects/ASEProject/EA/results/THP/BEST_POSCARS/best_gfnff.vasp', format='vasp')
    #
    # nuevo.calc = calc
    # # print(nuevo.get_potential_energy())
    # # check_uspex(parallel=False)
    # # check_uspex(parallel=False)
    # # check_uspex(parallel=False)
    # # check_uspex(parallel=False)
    # # check_uspex(parallel=False)
    # # check_uspex(parallel=False)
    # dan = RelaxConfig
    # run(dan , nuevo, Path('/home/vito/PythonProjects/ASEProject/EA/results/THP/BEST_POSCARS'))

    data = get_structure_from_id(poscar_dir='/home/vito/PythonProjects/ASEProject/EA/results/THP/BEST_POSCARS/collected_theophylline_2POSCARS',
                                id_structures=[1], all=True)
    batch = []
    for k, v in data.items():
        atom = read(StringIO(v), format='vasp')
        batch.append(atom)

    from ase.io import Trajectory
    traj = Trajectory("/home/vito/Documents/USPEX_test/004_theophylline_4_mol_gulp_trained/results1/structuresmy2.traj", "w")

    # Write all structures
    for atoms in batch:
        traj.write(atoms)

    traj.close()

    end2 = time.perf_counter()
    print(f"Elapsed time: {end2 - start2} seconds")
