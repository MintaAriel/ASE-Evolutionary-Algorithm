from pymatgen.core import Molecule as PymatgenMolecule
from pyxtal import pyxtal
from pyxtal.msg import Comp_CompatibilityError
import numpy as np
from ase.io import write
import os
import random

def ase_to_pymatgen(ase_atoms):
    symbols = [atom.symbol for atom in ase_atoms]
    coords = ase_atoms.positions
    return PymatgenMolecule(symbols, coords)

class create_pyxtal():
    def __init__(self, unique=False):
        self.unique = unique
        self.groups = [i for i in range(2,231)]
        self.allowed_groups = []


    def sym_molcrys( self, molecule, n):
        while True:
            my_mol = ase_to_pymatgen(molecule)

            try:
                xtal = pyxtal(molecular=True)
                if len(self.groups) > 0:
                    sym_group_idx = random.randrange(len(self.groups))
                    sym_group = self.groups[sym_group_idx]
                    self.groups.pop(sym_group_idx)
                    if len(self.groups) == 1:
                        print(f'\n Max amount of spacegroups achieved: {len(self.allowed_groups)}\n'
                              f' == No more unique symmetries, switching to non-unique == \n\n')

                else:
                    sym_group_idx = random.randrange(len(self.allowed_groups))
                    sym_group = self.allowed_groups[sym_group_idx]
                # print(f'Trying spacegroup {sym_group}')
                success = xtal.from_random(
                    dim=3,
                    group=sym_group,
                    species=[my_mol],  # Use 'species' instead of 'molecules'
                    numIons=[n],  # Use 'numIons' instead of 'numMols'
                    factor=1.1,
                    max_count=1
                )
                pyxtal_mol = xtal.to_ase(resort=False)


                if success:
                    print(f"Molecular crystal generated successfully, symmetry {sym_group}!")
                    if len(self.groups) > 0:
                        self.allowed_groups.append(sym_group)

                    pyxtal_mol.info['spacegroup'] = sym_group


                return pyxtal_mol

            except Exception as e:
                # print("Failed to generate a valid crystal structure.", e)

                continue

def sym_molcrys_unique( molecule, n, spacegroup):
    my_mol = ase_to_pymatgen(molecule)

    while True:

        try:

            print(f"Trying spacegroup {spacegroup}")

            xtal = pyxtal(molecular=True)

            success = xtal.from_random(
                dim=3,
                group=spacegroup,
                species=[my_mol],
                numIons=[n],
                factor=1.1
            )

            if success:
                pyxtal_mol = xtal.to_ase(resort=False)
                pyxtal_mol.info['spacegroup'] = spacegroup
                print(f"Generated successfully with symmetry {spacegroup}")
                return pyxtal_mol
        except:

            # print("Failed to generate a valid crystal structure.")
            continue

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
        'This method can be used to create seed for USPEX using pyxtal'
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


# my_mol = Molecule(mol_path) # Use this if you have a file

# urea_mol = mol2ase( MOL_path='/home/vito/PythonProjects/ASEProject/EA/MOLCRYS/test_urea/MOL_1')
# urea = urea_mol.read()
#
# atom = urea

#
# initial_pop = pyxtal_to_poscar(blocks = [('Mg', 4), ('Al',8),('O', 16)],
#                                amount=40,
#                                directory='/home/brian/PycharmProjects/ASEProject/uspex_python/Seeds')
#initial_pop.create_poscars()
