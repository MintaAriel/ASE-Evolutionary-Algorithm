import spglib
from ase.io import read, write
from ase.visualize import view
from ase import Atoms


# experimental_2 =  read('/home/vito/PythonProjects/ASEProject/CARLO/gnff/Fine_tun/exp_2_MUT_2.cif')
experimental_2 = read('/home/vito/PythonProjects/ASEProject/EA/results/THP/optd/str_29_optd_pbebj_9a.vasp')


def get_prim(structure, out_dir):
    atoms = structure * (2, 2, 2)
    cell = (
        atoms.cell,
        atoms.get_scaled_positions(),
        atoms.numbers
    )

    primitive = spglib.find_primitive(cell, symprec=1e-4)

    lattice, positions, numbers = primitive

    prim_atoms = Atoms(
        numbers=numbers,
        scaled_positions=positions,
        cell=lattice,
        pbc=True
    )
    print(prim_atoms)
    write(out_dir, prim_atoms, format='cif')

    print(primitive)

    # view(prim_atoms)


# experimental = read('/home/vito/PythonProjects/ASEProject/EA/results/THP/optd/str_20_optd_pbebj_9a.vasp')
experimental = read('/home/vito/PythonProjects/ASEProject/EA/results/THP/BEST_POSCARS/476POSCAR', format='vasp')


get_prim(experimental, out_dir='/home/vito/PythonProjects/ASEProject/EA/results/THP/polyclean_bj/476.cif')
# get_prim(experimental_2,out_dir='/home/vito/PythonProjects/ASEProject/EA/results/THP/polyclean_bj/29.cif')

