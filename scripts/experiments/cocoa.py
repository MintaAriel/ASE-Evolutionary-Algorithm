
from ase.io import read
from pygulp.molecule.fix_mol_gradient import define_ASU, generate_tags, transformASU
from ase.visualize import view
import json
import pandas as pd


poly5 = read('/home/vito/Downloads/629822.cif', format='cif')

print(poly5)

tags = generate_tags(n_atoms=171, n_molecules=4, mode='cycle')
poly5.set_tags(tags)
print(tags)

mask = poly5.get_tags() == 1
atoms_tag1 = poly5[mask]



asu, spgr, n_mol = define_ASU(poly5, n_mol=4)

print(n_mol)

# view(poly5)
view(asu)

# recon = transformASU(asu, sym_group=9)
#
# new = recon.get_full_sym_cell(asu)
# print(new)
#
# # view(new)

from ase.io import write

mol = asu.copy()
mol.set_pbc(False)

view(mol)