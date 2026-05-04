
from ase.io import read, write
from sympy.physics.units import length

from ea.io.read_con import parse_connections2
from pygulp.molecule.fix_mol_gradient import define_ASU, generate_tags, transformASU
from ase.visualize import view
from ase.io import Trajectory
from ea.io.create_mol import create_mol_from_asu, make_mol_file
from ase import Atoms
from ea.structures.ase_creator import NIH2ase, mol2ase, mol2ase2
import numpy as np
import json
import pandas as pd


def make_asu():
    poly5 = read('/home/vito/Downloads/dr5011sup1.cif', format='cif')

    print(poly5)

    tags = generate_tags(n_atoms=171, n_molecules=4, mode='cycle')
    poly5.set_tags(tags)
    print(tags)

    mask = poly5.get_tags() == 1
    atoms_tag1 = poly5[mask]

    asu, spgr, n_mol = define_ASU(poly5, n_mol=4)

    print(n_mol)

    view(asu)
    w,e,r,f = create_mol_from_asu(asu, cov_radii=0.9)
    make_mol_file(e, asu, r, f, '/home/vito/PythonProjects/ASEProject/EA/data/SOS')
    print(r)
#
# make_asu()

a = mol2ase2('/home/vito/PythonProjects/ASEProject/EA/data/SOS/b1molecule.mol')
b = mol2ase2('/home/vito/PythonProjects/ASEProject/EA/data/SOS/b2molecule.mol')
#
view(a)
view(b)

# w,e,r,f = create_mol_from_asu(a, cov_radii=0.9)
# make_mol_file(e, a, r, f, '/home/vito/PythonProjects/ASEProject/EA/data/SOS')
# print(r)

from ea.core.mutation import mutate_reaxff, mutate_reaxff_small, rearange_mut
from ea.structures.create_seeds import write_POSCAR

# d = write_POSCAR('/home/vito/uspex_matlab/SOS/test1/theophilline.db', '/home/vito/uspex_matlab/SOS/test1')
# d.create(20)
# mut = mutate_reaxff_small('/home/vito/uspex_matlab/SOS/test1', '/home/vito/uspex_matlab/SOS/test1/connections')
# mut.mutate(20,  '4_POSCARS', factor=0.75)



# se = mut.da.get_atoms(2)
# lengths = [8.211, 5.4624,  65.38, ]
#
# # print(se.cell[0][0])
# print(se.cell)
# view(se)
# # print(se.get_tags())
# new = rearange_mut('/home/vito/uspex_matlab/SOS/test1',
#                    '/home/vito/uspex_matlab/SOS/test1/connections',
#                    se, lengths)
#
# print(new)
# view(new)
# write('/home/vito/uspex_matlab/SOS/test1/exp3.cif', new, format='cif')
# parse_connections2('/home/vito/PythonProjects/ASEProject/EA/data/SOS/molecule.mol')