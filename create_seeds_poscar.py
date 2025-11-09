from pyxtal import pyxtal
from pyxtal.msg import Comp_CompatibilityError
import numpy as np
from ase import Atoms
from ase.io import write
from sympy.physics.units import amount
import os

class pyxtal_to_poscar():

    def __init__(self, blocks, amount, directory):
        self.blocks = blocks
        self.amount = amount
        self.directory = directory
        self.file_name = self.directory + '/POSCARS_1'

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
            pos = write(self.directory+'/POSCAR', molecule, format='vasp')

            # Read the POSCAR content and append to a master file
            with open(self.directory+'/POSCAR', 'r') as f:
                content = f.read()

            with open(self.file_name, 'a') as master_file:
                #master_file.write(f"number={i + 1} \n")
                master_file.write(content)

            print(f"Structure {i + 1} written and appended")

        os.remove(self.directory+'/POSCAR')


#
# initial_pop = pyxtal_to_poscar(blocks = [('Mg', 4), ('Al',8),('O', 16)],
#                                amount=40,
#                                directory='/home/brian/PycharmProjects/ASEProject/uspex_python/Seeds')
#initial_pop.create_poscars()
