
from ase import Atoms
import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import numpy as np
from pathlib import Path
from deepmd.calculator import DP
from ase.io import write
from ase.io import read


work_dir = Path('/home/vito/PythonProjects/ASEProject/EA/results/THP/Grant/phonons/ML/128707/1_3_2')

calc = DP(model='/home/vito/PythonProjects/ASEProject/EA/models/dpa3-pbed3-pytorch.pth', device='gpu')


atom = read(work_dir/ '103.vasp', format='vasp')
# atom.calc = calc

def generate_displacements(atoms, supercell=(1,2,1), distance=0.01):

    unitcell = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(),
    )

    phonon = Phonopy(
        unitcell,
        supercell_matrix=np.diag(supercell)
    )

    phonon.generate_displacements(distance=distance)

    phonon.save( work_dir / "phonopy_disp.yaml")

    return phonon

phonon = generate_displacements(atom)


forces = []

for scell in phonon.supercells_with_displacements:

    ase_atoms = Atoms(
        symbols=scell.symbols,
        cell=scell.cell,
        scaled_positions=scell.scaled_positions,
        pbc=True,
    )

    ase_atoms.calc = calc

    forces.append(ase_atoms.get_forces())

from phonopy.file_IO import write_FORCE_SETS

dataset = phonon.dataset

write_FORCE_SETS(dataset, filename=work_dir / "FORCE_SETS")