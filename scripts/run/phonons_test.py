from pygulp.relaxation.relax import Gulp_relaxation_noadd, MyGULP
from ase.io import read
from ase.calculators.gulp import Conditions
from ase.visualize import view
import os
import numpy as np
from pygulp.molecule.fix_mol_gradient import define_ASU


crystal = read('/home/vito/PythonProjects/ASEProject/container_gpu_2 (2)/structures/1039798.cif')


out_dir = '/home/vito/PythonProjects/ASEProject/EA/test/phonopy/gfnff'

gulp_input = (f"opti gradient conp conse qok c6 conp prop gfnff gwolf noauto\n"
                  f"gfnff_scale 0.8 1.343 0.727 1.0 2.859\n"
                  f"maths mrrr\n"
                  f"pressure 0 GPa"
                  )


gulp_input_no_relax = (f"gradient conp conse qok c6 conp prop gfnff gwolf noauto\n"
                  f"gfnff_scale 0.8 1.343 0.727 1.0 2.859\n"
                  f"maths mrrr\n"
                  f"pressure 0 GPa"
                  )
options = (
    "output movie cif out1.cif\n"
    "maxcycle 300\n"
    "gtol 0.00001"
)



relax = Gulp_relaxation_noadd(path=out_dir,
                              library=None,
                              gulp_keywords=gulp_input,
                              gulp_options=options)

new_crys = relax.use_gulp(crystal)
# forces = new_crys.get_forces()
# print(forces)
# view(new_crys)

calc = MyGULP(keywords=gulp_input_no_relax,
                                # goutput file parameters from USPEX
                                options=[options],
                                # maybe Optional Parameters
                                library=None,
                                conditions=None)

calc.directory = os.path.join(out_dir, 'CalcFold')


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

print("Starting force-constants calculation...")

supercell = [2, 2, 2]  # supercell matrix (diag)
displacement = 0.02
symprec = 1e-3


asu = define_ASU(crystal, tolerance =symprec)

phonon = calculate_force_constants(
    calculator=calc,
    atoms=new_crys,
    supercell_matrix=supercell,
    displacement=displacement,
    symprec=symprec,
)
phonon.save(settings={"force_constants": True})
print(phonon)
