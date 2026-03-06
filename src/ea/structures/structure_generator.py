from ase import Atoms
from ase.ga.data import PrepareDB
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import CellBounds, closest_distances_generator
import numpy as np
from .pyxtal_ea import create_pyxtal


def count_stoichometry(lst):
    """
    Count occurrences of each unique value in a list.

    Args:
        lst (list): Input list of hashable elements.

    Returns:
        tuple: (counts_dict, unique_tuple)
            counts_dict: Dictionary with elements as keys and their counts as values.
            unique_tuple: Tuple containing the unique elements in the order they first appear.
    """
    counts = {}
    unique_list = []
    for item in lst:
        if item not in counts:
            counts[item] = 1
            unique_list.append(item)
        else:
            counts[item] += 1
    return counts, tuple(unique_list)


class first_gen_mol():
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

    def __init__(self, blocks, population, volume, db_path, symmetry, splits = None):
        self.blocks = blocks
        self.population = population
        self.volume = volume
        self.splits = splits
        self.db_path = db_path
        self.symmetry = symmetry
        # If symmetry is False, splits must be provided
        if self.symmetry is False and self.splits is None:
            raise ValueError('When symmetry is False, splits must be provided')

    def create_structures(self):

        molecule = self.blocks[0][0]
        n_mol = self.blocks[0][1]
        atom_num = molecule.get_atomic_numbers()
        sym = molecule.get_chemical_symbols()* n_mol
        stoichiometry, _ = count_stoichometry(sym)

        blmin = closest_distances_generator(
            atom_numbers=atom_num, ratio_of_covalent_radii=1
        )

        # Specify reasonable bounds on the minimal and maximal
        # cell vector lengths (in angstrom) and angles (in degrees)
        cellbounds = CellBounds(
            bounds={
                'phi': [15, 150],
                'chi': [15, 150],
                'psi': [15, 150],
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

        da = PrepareDB(db_file_name=self.db_path, simulation_cell= slab, stoichiometry=stoichiometry)

        if self.symmetry == True:
            pyx_generator = create_pyxtal(unique=True)

        # Generate N random structures and add them to the database
        for i in range(self.population):

            if self.symmetry == True:
                # Create tags for structures created in Pyxtal

                tags = [i for i in range(n_mol) for _ in range(len(molecule))]

                #Create charges
                charge = molecule.get_initial_charges()
                charges = np.tile(charge, n_mol)


                #Not unique
                # a = sym_molcrys(molecule, n_mol)
                a = pyx_generator.sym_molcrys(molecule, n_mol)
                a.set_tags(tags)
                a.set_initial_charges(charges)
                da.add_unrelaxed_candidate(a, spacegroup = a.info['spacegroup'])

            elif self.symmetry == False:
                a = None
                counter = 0
                while a == None:
                    try:
                        a = sg.get_new_candidate(maxiter=30)
                    except Exception as e:
                        print('ASE internal error: ',e)
                    counter +=1

                da.add_unrelaxed_candidate(a)





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


class first_gen_atomic():
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

    def sym_random_generator(self, block):
        elements, counts = zip(*block)
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
                atom.symbols = ''.join(f'{el}{count}' for el, count in block)
                atom.info['spacegroup'] = sym_group
                return atom
            except Comp_CompatibilityError as e:
                print('error')
                continue



    def create_structures(self):

        # Generate a dictionary with the closest allowed interatomic distances(Mg-Mg, Mg-O, O-O)
        atom_num = [atomic_numbers[name] for (name, number) in self.blocks]
        print(atom_num)

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
        for v in self.block_guess:
            print(v, type(v))



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
        f = self.blocks
        sg = StartGenerator(
            slab,
            self.block_guess[0],
            blmin,
            box_volume=self.volume_guess[0],
            number_of_variable_cell_vectors=3,
            cellbounds=cellbounds,
            splits=self.splits,
        )

        print(sg.blocks, type(sg.blocks[0][0]))
        print(cellbounds, 'This is cellbounds')

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
                    a = self.sym_random_generator(self.block_guess[i])
                    da.add_unrelaxed_candidate(a, spacegroup = a.info['spacegroup'])
                elif self.symmetry == False:
                    sg = StartGenerator(
                                        slab,
                                        self.block_guess[i],
                                        blmin,
                                        box_volume=self.volume_guess[i],
                                        number_of_variable_cell_vectors=3,
                                        cellbounds=cellbounds,
                                        splits=self.splits,
                                    )
                    a = sg.get_new_candidate()
                    da.add_unrelaxed_candidate(a)






