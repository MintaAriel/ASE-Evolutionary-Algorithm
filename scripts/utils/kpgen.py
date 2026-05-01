from ase.io import read
from ase.calculators.vasp import Vasp

atoms = read("/home/vito/PythonProjects/ASEProject/from_Carlo_to_Brian/str_6_optd_9a.vasp", format='vasp')

calc = Vasp(kspacing=0.3, kgamma=True)
kpts = calc.get_kpt(atoms)

print(kpts)