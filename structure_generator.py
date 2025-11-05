from ase import Atoms
from ase.data import atomic_numbers
from ase.ga.data import PrepareDB
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import CellBounds, closest_distances_generator
from ase.ga.data import DataConnection
from ase.visualize import view
from ase.io import write
from pyxtal import pyxtal
from pyxtal.msg import Comp_CompatibilityError
import numpy as np
from ase.spacegroup import get_spacegroup
import spglib
import sqlite3


class generation_generator():
    '''
    blocks: list
        Specify the 'building blocks' from which the initial structures
        will be constructed. We will put the atoms next to their coefficient in a tuple
        Mg4Al8O16 into [('Mg', 4), ('Al',8),('O', 16)]
    population: int
        Number of random initial structures to generate
    volume:
        Target cell volume for the initial structures, in angstrom^3
    splits:
        Choose an (optional) 'cell splitting' scheme which basically
        controls the level of translational symmetry (within the unit
        cell) of the randomly generated structures. Here a 1:1 ratio
        of splitting factors 2 and 1 is used:
        splits = {(4,): 1, (1,): 1}
        There will hence be a 50% probability that a candidate
        is constructed by repeating 4 times the structure along a
        randomly chosen axis. In the other 50%
        of cases, no cell cell splitting will be applied.
    db_name: str
        name of the db file to store the structures
    '''

    def __init__(self, blocks, population, volume, splits, db_name, symmetry):
        self.blocks = blocks
        self.population = population
        self.volume = volume
        self.splits = splits
        self.db_name = db_name
        self.symmetry = symmetry

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

    def create_structures(self):

        # Generate a dictionary with the closest allowed interatomic distances(Mg-Mg, Mg-O, O-O)
        atom_num = [atomic_numbers[name] for (name, number) in self.blocks]

        # stoichiometry is a list of the atomic numbers of all the atoms, it should be the same length
        # as numer of atoms in the cell
        stoichiometry = [atomic_numbers[name] for name, count in self.blocks for _ in range(count)]

        blmin = closest_distances_generator(
            atom_numbers=atom_num, ratio_of_covalent_radii=0.5
        )

        # Specify reasonable bounds on the minimal and maximal
        # cell vector lengths (in angstrom) and angles (in degrees)
        cellbounds = CellBounds(
            bounds={
                'phi': [35, 145],
                'chi': [35, 145],
                'psi': [35, 145],
                'a': [3, 50],
                'b': [3, 50],
                'c': [3, 50],
            }
        )

        # The 'slab' object in the GA serves as a template
        # in the creation of new structures, which inherit
        # the slab's atomic positions (if any), cell vectors
        # (if specified), and periodic boundary conditions.
        # Here only the last property is relevant:
        slab = Atoms('', pbc=True)

        # Initialize the random structure generator
        sg = StartGenerator(
            slab,
            self.blocks,
            blmin,
            box_volume=self.volume,
            number_of_variable_cell_vectors=3,
            cellbounds=cellbounds,
            splits=self.splits,
        )

        da = PrepareDB(db_file_name=self.db_name, simulation_cell= slab, stoichiometry=stoichiometry)
        #add a column for spacegoup
        '''
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('ALTER TABLE systems ADD COLUMN spacegroup INTEGER ')
        conn.commit()
        conn.close()
        '''

        # Generate N random structures
        # and add them to the database
        for i in range(self.population):
            if self.symmetry == True:
                a = self.sym_random_generator()
                da.add_unrelaxed_candidate(a, spacegroup = a.info['spacegroup'])
            elif self.symmetry == False:
                a = sg.get_new_candidate()
                da.add_unrelaxed_candidate(a)

class population_vis():
    def __init__(self, db_name, num_prev):
        self.db_name = db_name
        self.num_prev = num_prev
        self.da = DataConnection(self.db_name)

    def open_one_struc(self, i):
        slab = self.da.get_atoms(i)
        #print(slab.data['parents'])
        sg = spglib.get_spacegroup((slab.get_cell(), slab.get_scaled_positions(),
                                    slab.get_atomic_numbers()),
                                   symprec=1e-2)
        if sg is None:
            raise RuntimeError('Spacegroup not found')
        sg_no = int(sg[sg.find('(') + 1:sg.find(')')])
        print(sg)
        print(sg_no)
        try:
            print(slab.info['spacegroup'])
        except:
            pass
        print(slab.get_cell(), '\n', slab.get_scaled_positions(), '\n')
        print(slab.get_cell().lengths())
        view(slab)

    def open_structures(self):
        for i in range(2, self.num_prev+ 2):
            slab = self.da.get_atoms(i)
            view(slab)

    def get_pairs(self, id):
        traj = self.da.get_all_relaxed_candidates()
        slab = self.da.get_atoms(id)
        print(slab)
        view(slab)
        print(traj)
        for v in traj:
            if v.info['key_value_pairs']['gaid'] == id:
                print(v)
                print(v.info['key_value_pairs'])
                view(v)


    def plot_structures(self):

        for i in range(22, self.num_prev + 2):
            atom = self.da.get_atoms(i)
            write('plots_db/front.png', atom, rotation='70x, 45y, 0z')
            write('plots_db/side.png', atom, rotation='70x, 90y, 0z')



gen = generation_generator(blocks = [('Mg', 4), ('Al',8),('O', 16)],
                           population=40,
                           volume = 240,
                           splits = {(4,): 1, (1,): 1},
                           db_name = 'GA1_sym_40.db',
                           symmetry = True)

gen2 = generation_generator(blocks = [('Mg', 4), ('Al',8),('O', 16)],
                           population=40,
                           volume = 240.0,
                           splits = {(2,): 1, (1,): 1},
                           db_name = 'GA1_40.db',
                           symmetry=False)

#gen2.create_structures()


prev = population_vis('/home/vito/PythonProjects/ASEProject/EA/Programms/GA1_40.db', 5)
#prev_2 = population_vis('prueba4.db', 1)

#for i in range(1490,1495):
#    prev.open_one_struc(i)

prev.open_one_struc(15)
#prev.open_one_struc(155)
#prev.open_one_struc(135)

#prev.get_pairs(6)


