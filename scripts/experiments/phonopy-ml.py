from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from deepmd.calculator import DP

from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.file_IO import write_FORCE_SETS

# ============================================================
# Settings
# ============================================================

work_dir = Path(
    "/home/vito/PythonProjects/ASEProject/EA/results/THP/Grant/phonons/ML/128707/1_3_2"
)
# ============================================================
# Read structure
# ============================================================

atoms = read(work_dir / "128707.vasp")

model_path = (
    "/home/vito/PythonProjects/ASEProject/EA/models/dpa3-pbed3-pytorch.pth"
)

calc = DP(model=model_path, device="gpu")



# ============================================================
# Create Phonopy object
# ============================================================

unitcell = PhonopyAtoms(
    symbols=atoms.get_chemical_symbols(),
    cell=atoms.cell.array,
    scaled_positions=atoms.get_scaled_positions(),
)

phonon = Phonopy(
    unitcell,
    supercell_matrix=np.diag([1, 1, 1]),
)

phonon.generate_displacements(distance=0.01)

phonon.save(work_dir / "phonopy_disp.yaml")

# ============================================================
# Calculate forces
# ============================================================

forces = []

for i, scell in enumerate(phonon.supercells_with_displacements):

    ase_atoms = Atoms(
        symbols=scell.symbols,
        cell=scell.cell,
        scaled_positions=scell.scaled_positions,
        pbc=True,
    )

    ase_atoms.calc = calc

    f = ase_atoms.get_forces()

    print(
        f"Displacement {i+1}/{len(phonon.supercells_with_displacements)} "
        f"Force shape = {f.shape}"
    )

    forces.append(f)

# ============================================================
# Attach forces to dataset
# ============================================================

dataset = phonon.dataset

for disp, force in zip(dataset["first_atoms"], forces):
    disp["forces"] = force

# ============================================================
# Write FORCE_SETS
# ============================================================

write_FORCE_SETS(
    dataset,
    filename=work_dir / "FORCE_SETS"
)

print("FORCE_SETS written successfully")