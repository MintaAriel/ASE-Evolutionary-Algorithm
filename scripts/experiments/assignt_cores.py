structures_id = list(range(13, 41))
# CPU core affinity per structure: structure_id -> list of core IDs
core_affinity = {
    k: [2*(k-1), 2*(k-1) + 1]
    for k in structures_id
}

print(core_affinity)

from pygulp.molecule.fix_mol_gradient import define_ASU
from ase.io import read
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

mol_20 = read('/home/vito/PythonProjects/ASEProject/EA/data/theophylline/vasp-carlo/structure_20_optd_b.vasp')
poly86 = read('/home/vito/PythonProjects/ASEProject/EA/data/theophylline/vasp-carlo/862238_optd_b.vasp')
poly12 = read('/home/vito/PythonProjects/ASEProject/EA/data/theophylline/vasp-carlo/128707_optd_b.vasp')
from pymatgen.io.ase import AseAtomsAdaptor

def define_sym(crystal, tolerance = 0.01):
    pmg_structure = AseAtomsAdaptor.get_structure(crystal)
    # Assuming you have a pymatgen Structure object
    analyzer = SpacegroupAnalyzer(pmg_structure, symprec=tolerance, angle_tolerance=0.5)
    spacegroup = analyzer.get_symmetry_dataset().number
    print('spacegroup: ', spacegroup)


asu = define_sym(mol_20, 0.5e-2)
asu86 = define_sym(poly86, 0.5e-2)
asu12 = define_sym(poly12, 0.5e-2)

print(asu, asu86, asu12)