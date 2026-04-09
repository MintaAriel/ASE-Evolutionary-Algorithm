from ase.io import read
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.visualize import view
from pygulp.molecule.fix_mol_gradient import define_ASU

force = 0.1* np.array([
    [-110.922561,  -397.688659,  186.273103],  # 1 O
    [ -133.543317,   -34.208808,    42.929117],  # 2 N
    [ -162.902872,    96.861777,  -458.563124],  # 3 N
    [ -211.491238,   128.393582,    64.518749],  # 4 C
    [ -206.674431,   145.036581,   -13.459445],  # 5 C
    [   80.982725,    11.573398,   121.999353],  # 6 C
    [   93.366806,    47.491483,   -93.690593],  # 7 C
    [  -94.115393,  -122.755236,   -61.977942],  # 8 C
    [   56.137927,  -130.495664,   -76.606618],  # 9 C
    [   15.363494,    -1.246032,    83.808539],  # 10 C
    [   34.283020,    27.502931,   -74.289320],  # 11 C
    [   69.925033,    -3.400102,   222.378379],  # 12 C
    [   71.148691,    84.001637,  -159.407767],  # 13 C
    [ 24.600627,   38.894046, -101.642599],  # 14 C
    [  -56.806070,    43.007845,   -20.805355],  # 15 C
    [  -65.172525,    13.051242,    45.291955],  # 16 C
    [   67.298338,   -67.485562,   -48.803711],  # 17 C
    [   33.407156,  -101.371914,    26.922375],  # 18 C
    [  -51.086281,     4.215722,    38.320024],  # 19 H
    [   -6.745766,   -16.767911,   -70.087972],  # 20 H
    [  -63.215621,    31.592569,   -13.289416],  # 21 H
    [  -60.372462,    24.459506,    25.008322],  # 22 H
    [   52.257258,   -34.631226,     2.451067],  # 23 H
    [   47.059859,   -35.202387,   -15.644444],  # 24 H
    [  -27.393306,    37.852407,    65.867109],  # 25 H
    [  -15.773231,    37.628397,   -39.801834],  # 26 H
    [   34.868151,    -9.756213,    72.665889],  # 27 H
    [   36.430575,    29.963587,   -73.252661],  # 28 H
    [  -51.005533,    -6.281007,   -29.405243],  # 29 H
    [   18.603815,     2.783004,   -53.889650],  # 30 H
])

forces =  np.repeat(force, 4,  axis=0)
atoms = read('/home/vito/Downloads/carba.cif', format='cif')
atoms.set_scaled_positions(atoms.get_scaled_positions()+[0.1,0.12,0.08])

# attach to atoms
atoms.calc = SinglePointCalculator(atoms, forces=forces)

# now ASE "knows" the forces
print(atoms.get_forces())
tags = np.tile([1, 2, 3, 4], 30)
mask = np.isin(tags, 1)
print(tags)
print(mask)
asu = atoms.__getitem__(mask)
asu.calc = SinglePointCalculator(asu, forces=force)
print(asu.get_forces())
print(asu)
view(asu)
view(atoms)
# asu, spacegroup, _ = define_ASU(atoms)
# print(asu)
# view(asu)

# view(atoms)

