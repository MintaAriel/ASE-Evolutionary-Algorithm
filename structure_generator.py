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

def get_volumne(num_atoms):
    'as shown in ase, an initial guess are values of 8-12 A^3 per atom'
    volume = (np.random.randint(80, 121) / 10.0) * num_atoms
    return volume

def random_sum_uniform(target_sum, n_numbers):
    # Generate (n_numbers - 1) cut points between 0 and target_sum
    cuts = np.sort(np.random.choice(range(1, target_sum), n_numbers - 1, replace=False))
    parts = np.diff([0, *cuts, target_sum])
    return parts.tolist()


def random_block(min, max, block, build_blocks):
    max = max+1
    for v in build_blocks:
        if len(v) != len(block):
            raise ValueError(f'Error: building blocks array doesnt match with the number of building atoms ({len(block)})')

    #num_atoms = np.random.randint(min, max)
    num_elements = len(block)

    num_atom_in_build = np.array([sum(sublist) for sublist in build_blocks])
    if sum(num_atom_in_build) > max:
        raise ValueError(f'Error: building blocks contains more atoms that allowed by maxAt ({max})')

    stoich = 0
    while True:
        num_atoms = np.random.randint(min, max)
        ratio = random_sum_uniform(num_atoms, len(build_blocks))
        floor = ratio//num_atom_in_build
        count = 0
        if 0 in floor:
            while count < len(build_blocks):
                ratio = np.roll(ratio, 1).tolist()
                #print(ratio)
                count += 1
                floor = ratio // num_atom_in_build

                if 0 not in floor:
                    break

            stoich = floor * num_atom_in_build
        else:
            stoich = floor * num_atom_in_build

        #print(stoich)
        #print(num_atoms)
        if sum(stoich)>=min and sum(stoich)<=max and 0 not in floor:
            break

    #print(floor, 'This is the end')
    return floor


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
    varcomp: list
        a list with the min amount and max amount of atoms [minAt, maxAt]
    build_blocks: matrix
        matrix of compositional building blocks for variable composition
        as used in Uspex (numSpecies) manual pg. 39
        [[2, 0],
        [0, 1]]
        For a system AB that means that the building blocks are 2A and B
    '''

    def __init__(self, blocks, population, volume, splits, db_name, symmetry, varcomp, build_blocks = None):
        self.blocks = blocks
        self.block_guess = None
        self.population = population
        self.volume = volume
        self.volume_guess = None
        self.splits = splits
        self.db_name = db_name
        self.symmetry = symmetry
        self.varcomp = varcomp
        self.build_blocks = build_blocks

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

        if self.varcomp != None:
            minAt = min(self.varcomp)
            maxAt = max(self.varcomp)
            element_name = [t[0] for t in self.blocks]
            "def random_block generates random variable composition, we store this in another list"
            self.block_guess = [list(zip(element_name,random_block(minAt,maxAt, self.blocks, self.build_blocks))) for _ in range(self.population)]
            self.volume_guess = [get_volumne(sum(value for _, value in sublist)) for sublist in self.block_guess]
        else:
            if self.volume == None:
                self.volume_guess = [get_volumne(len(stoichiometry)) for i in range(self.population)]

            else:
                self.volume_guess = [self.volume]

        print(self.volume_guess)
        print(self.block_guess)



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
        print(type(self.blocks))
        print(self.blocks)
        sg = StartGenerator(
            slab,
            blmin,
            blocks=self.blocks,
            box_volume=self.volume_guess[0],
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
            if self.varcomp == None:
                if self.symmetry == True:
                    a = self.sym_random_generator()
                    da.add_unrelaxed_candidate(a, spacegroup = a.info['spacegroup'])
                elif self.symmetry == False:
                    a = sg.get_new_candidate()
                    if len(self.volume_guess) > 1:
                        sg.box_volume = self.volume_guess[i]
                    else:
                        None
                    da.add_unrelaxed_candidate(a)
            else:
                if self.symmetry == True:
                    a = self.sym_random_generator()
                    da.add_unrelaxed_candidate(a, spacegroup = a.info['spacegroup'])
                elif self.symmetry == False:
                    a = sg.get_new_candidate()
                    if len(self.volume_guess) > 1:
                        sg.box_volume = self.volume_guess[i]
                        sg.blocks = self.block_guess[i]
                    else:
                        None
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



gen_spinel = generation_generator(blocks = [('Mg', 4), ('Al',8),('O', 16)],
                           population=40,
                           volume = 240,
                           splits = {(4,): 1, (1,): 1},
                           db_name = 'GA1_sym_40.db',
                           symmetry = True,
                           varcomp=None)

gen_spinel2 = generation_generator(blocks = [('Mg', 4), ('Al',8),('O', 16)],
                           population=10,
                           volume = None,
                           splits = {(2,): 1, (1,): 1},
                           db_name = 'GA1_40.db',
                           symmetry=False,
                           varcomp=None,)

gen_MoB = generation_generator(blocks = [('Mo', 1), ('B',1)],
                           population=20,
                           volume = 240.0,
                           splits = {(2,): 1, (1,): 1},
                           db_name = 'MoB.db',
                           symmetry=False,
                           varcomp=[8,18],
                            build_blocks=[[1,0],[0,1]])

gen_MoB.create_structures()

#
# prev = population_vis('/home/vito/PythonProjects/ASEProject/EA/Programms/GA1_40.db', 5)
#prev_2 = population_vis('prueba4.db', 1)

#for i in range(1490,1495):
#    prev.open_one_struc(i)
#
# prev.open_one_struc(15)
#prev.open_one_struc(155)
#prev.open_one_struc(135)

#prev.get_pairs(6)


