import time
import numpy as np

from ea.backends.deepmd_client import DeepMDClient
from ea.utils.config import load_config
from ase.io import read

from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms





def ase_to_phonopy_atoms(ase_atoms):
    """Convert ASE Atoms -> PhonopyAtoms."""
    return PhonopyAtoms(
        symbols=ase_atoms.get_chemical_symbols(),
        cell=ase_atoms.get_cell(),
        scaled_positions=ase_atoms.get_scaled_positions(),
    )


def phonopy_atoms_to_ase(patoms: PhonopyAtoms):
    """Convert PhonopyAtoms -> ASE Atoms."""
    from ase import Atoms
    return Atoms(
        symbols=patoms.symbols,
        cell=patoms.cell,
        scaled_positions=patoms.scaled_positions,
        pbc=True,
    )


def calculate_force_constants(calculator, atoms, supercell_matrix, displacement=0.01, symprec=1e-5):
    """
    Standard phonopy finite-displacement workflow (phonopy 2.48):
      - create Phonopy
      - generate_displacements
      - loop over phonon.supercells_with_displacements
      - compute forces
      - phonon.forces = sets_of_forces
      - produce_force_constants
    """
    unitcell = ase_to_phonopy_atoms(atoms)

    phonon = Phonopy(
        unitcell,
        supercell_matrix=supercell_matrix,
        symprec=symprec,
    )

    phonon.generate_displacements(distance=float(displacement))

    # phonopy 2.48 provides displaced supercells as a property (as in the official docs)
    scells = phonon.supercells_with_displacements
    if scells is None or len(scells) == 0:
        raise RuntimeError("No displaced supercells generated. Check supercell_matrix and symprec.")

    print(f"Number of displaced supercells: {len(scells)}")

    forces = []
    for i, sc in enumerate(scells, 1):
        sc_ase = phonopy_atoms_to_ase(sc)
        sc_ase.calc = calculator
        f = sc_ase.get_forces()
        forces.append(f)
        print(f"  forces done: {i}/{len(scells)}  shape={f.shape}")

    forces = np.asarray(forces, dtype=float)
    if forces.ndim != 3 or forces.shape[2] != 3:
        raise RuntimeError(f"Unexpected forces shape: {forces.shape}, expected (n_disp, n_atoms, 3)")

    phonon.forces = forces
    phonon.produce_force_constants()
    return phonon

cfg = load_config()
cif_path = '/home/vito/PythonProjects/ASEProject/container_gpu_2/Results/128707/base_deepmd/final.cif'

supercell = [1, 1, 1]
displacement = 0.02
symprec = 1e-3

for device in ("cuda", "cpu"):
    print(f"\n{'='*50}")
    print(f"  Device: {device}")
    print(f"{'='*50}")

    atom = read(cif_path)
    calc = DeepMDClient(config=cfg, device=device, mode="worker")

    # warm-up call (model loading, JIT, etc.)
    t0 = time.perf_counter()
    atom.calc = calc
    energy = atom.get_potential_energy()
    t_warmup = time.perf_counter() - t0
    print(f"Warm-up energy: {energy:.6f} eV  ({t_warmup:.2f}s)")

    # phonon benchmark
    t0 = time.perf_counter()
    phonon = calculate_force_constants(
        calculator=calc,
        atoms=atom,
        supercell_matrix=supercell,
        displacement=displacement,
        symprec=symprec,
    )
    t_phonon = time.perf_counter() - t0

    n_disp = len(phonon.supercells_with_displacements)
    print(f"Force constants: {n_disp} displacements in {t_phonon:.2f}s")
    print(f"  -> {t_phonon/n_disp:.3f} s/call  ({n_disp/t_phonon:.1f} calls/s)")

    calc.close()

    phonon.save(settings={f"force_constants": True}, filename=f'phonopy_params_{device}')
    print("\nForce constants saved.")
