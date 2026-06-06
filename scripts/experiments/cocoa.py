
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


# NIH2ase('/home/vito/PythonProjects/ASEProject/EA/data/theobromine/Conformer3D_COMPOUND_CID_5429.json',
#         '/home/vito/PythonProjects/ASEProject/EA/data/theobromine/')

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


# carba = read('/home/vito/PythonProjects/ASEProject/EA/data/carbamazepine/cif/carb2.cif')
#
# asu, _, _ = define_ASU(carba)
# view(asu)
# a = mol2ase2('/home/vito/PythonProjects/ASEProject/EA/data/SOS/b1molecule.mol')
# b = mol2ase('/home/vito/PythonProjects/ASEProject/EA/data/theobromine/MOL_1')
# c = b.read()
# print(c)
# #
#
# view(c)

# w,e,r,f = create_mol_from_asu(a, cov_radii=0.9)
# make_mol_file(e, a, r, f, '/home/vito/PythonProjects/ASEProject/EA/data/SOS')
# print(r)

from ea.core.mutation import mutate_reaxff, mutate_reaxff_small, rearange_mut
from ea.structures.create_seeds import write_POSCAR

# d = write_POSCAR('/home/vito/uspex_matlab/SOS/test1/theophilline.db', '/home/vito/uspex_matlab/SOS/test1')
# d.create(20)
mut = mutate_reaxff_small('/home/vito/uspex_matlab/Theo8/theo_4mol', '/home/vito/uspex_matlab/Theo8/test_2/connections')
mut.mutate(80,  '2_POSCARS', factor=0.75, keep_traj=True)



# se = mut.da.get_atoms(2)
# lengths = [8.211, 5.4624,  65.38, ]
#
# # print(se.cell[0][0])
# print(se.cell)
# view(se)
# # # print(se.get_tags())
# V0 = se.get_volume()
# new = rearange_mut('/home/vito/uspex_matlab/SOS/test1',
#                    '/home/vito/uspex_matlab/SOS/test1/connections',
#                    se, lengths)
#
# Vf = new.get_volume()
# view(new)
#
# print(Vf-V0)
# # view(new)
# tra = read('/home/vito/uspex_matlab/SOS/test1/gulp_sim.trajectory', index=':')[0]
# write('/home/vito/uspex_matlab/SOS/test1/first_POSCARS', tra, format='cif')
# parse_connections2('/home/vito/PythonProjects/ASEProject/EA/results/THP/polyclean_bj/38_pair_02_a0_n1_-110.mol')