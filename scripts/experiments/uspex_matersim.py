from ea.analysis.benchmark_mattersim import MatterSimTester
from ea.utils.config import load_config
from ase.atoms import Atoms
from ase.io import read, write
from ase.visualize import view
import shutil
from pathlib import Path
import os
import time

print("CWD at start:", os.getcwd())

def pick_input() -> str:
    """Using this comand in USPEX
    % abinitioCode
    99
    % ENDabinit

    When creating the calcl folder, there will be a poscar
    file named geom.in, this function reads the path of the
    file and transforms it into an ase ASE atoms object
    """
    if Path("geom.in").is_file():
        return read(Path("geom.in"), format='vasp')

    raise FileNotFoundError("Нет входного файла: geom.in")

atoms = pick_input()

cfg = load_config()
ms = cfg['mattersim']

tester = MatterSimTester(
    model_path=ms['model_path'],
    device=ms.get('device', 'cpu'),
    container_root=ms['container_root'],
    input_template=ms.get('input_template', 'input_mattersim_d3_short.py'),
    n_threads=ms.get('n_threads', 4),
)


# view(atoms)
print(atoms)

try:
    result = tester.relax(atoms, timeout=100)
    relaxed_atoms = result['relaxed_atoms']
    energy = result['final_energy']
    print('the relaxed structure', relaxed_atoms)
    print(result)
except Exception as e:
    print(f"Relaxation failed: {e}")
    print("Falling back to input structure with energy = 0")
    relaxed_atoms = atoms
    energy = 0

print("CWD before write:", os.getcwd())
print('saving as geom.out')

write('geom.out', relaxed_atoms, format='vasp', direct=True)

print('structure saved')

with open('energy.txt', 'w') as f:
    f.write(f"{energy}\n")