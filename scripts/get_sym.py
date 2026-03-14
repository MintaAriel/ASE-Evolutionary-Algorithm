from pygulp.molecule.fix_mol_gradient import define_ASU
from ase.io import read

best = read('/home/vito/PythonProjects/ASEProject/EA/test/matersim/best.cif')

_,_,_ = define_ASU(best, tolerance=0.1)