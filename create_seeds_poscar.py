from pyxtal import pyxtal
from pyxtal.msg import Comp_CompatibilityError
import numpy as np
from ase import Atoms
from ase.io import write
from sympy.physics.units import amount


class pyxtal_to_poscar():

    def __init__(self, blocks, amount, file_name):
        self.blocks = blocks
        self.amount = amount
        self.file_name = file_name

    def sym_random_generator(self):
        elements, counts = zip(*self.blocks)
        while True:
            xtal = pyxtal()
            try:
                sym_group = np.random.randint(2, 230)
                xtal.from_random(dim=3,
                                 group=sym_group,
                                 species=list(elements),
                                 numIons=list(counts),
                                 random_state=1)
                atom = xtal.to_ase()
                atom.symbols = ''.join(f'{el}{count}' for el, count in self.blocks)
                atom.info['spacegroup'] = sym_group
                return atom
            except Comp_CompatibilityError as e:
                print('error')
                continue

    def create_poscars(self):
        for i in range(self.amount):
            molecule = self.sym_random_generator()
            pos = write('POSCAR', molecule, format='vasp')

            # Read the POSCAR content and append to a master file
            with open('POSCAR', 'r') as f:
                content = f.read()

            with open(self.file_name, 'a') as master_file:
                master_file.write(f"number={i + 1} \n")
                master_file.write(content)

            print(f"Structure {i + 1} written and appended")



initial_pop = pyxtal_to_poscar(blocks = [('Mg', 4), ('Al',8),('O', 16)],
                               amount=3,
                               file_name='POSCARS_1')
initial_pop.create_poscars()
