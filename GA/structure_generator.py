from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols
from ase.ga.data import PrepareDB
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import CellBounds, closest_distances_generator
from ase.visualize import view

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
    '''

    def __init__(self, blocks, population, volume, splits, db_name):
        self.blocks = blocks
        self.population = population
        self.volume = volume
        self.splits = splits
        self.db_name = db_name

    def create_structures(self):

        # Specify the 'building blocks' from which the initial structures
        # will be constructed. We will put the atoms next to their coefficient in a tuple




        # Generate a dictionary with the closest allowed interatomic distances(Mg-Mg, Mg-O, O-O)
        atom_num = [atomic_numbers[name] for (name, number) in self.blocks]
        blmin = closest_distances_generator(
            atom_numbers=atom_num, ratio_of_covalent_radii=0.5
        )
        print(blmin)

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

        # Choose an (optional) 'cell splitting' scheme which basically
        # controls the level of translational symmetry (within the unit
        # cell) of the randomly generated structures. Here a 1:1 ratio
        # of splitting factors 2 and 1 is used:
        splits = {(4,): 1, (1,): 1}
        # There will hence be a 50% probability that a candidate
        # is constructed by repeating an randomly generated Ag12
        # structure along a randomly chosen axis. In the other 50%
        # of cases, no cell cell splitting will be applied.

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



        da = PrepareDB(db_file_name=self.db_name, stoichiometry=[Z1]*4+[Z2]*8+[Z3]*16)
        # Generate N random structures
        # and add them to the database
        for i in range(N):
            a = sg.get_new_candidate()
            da.add_unrelaxed_candidate(a)


        print(type(a))
        view(a)


gen = generation_generator(blocks = [('Mg', 4), ('Al',8),('O', 16)],
                           population=20,
                           volume = 240.0,
                           splits = {(4,): 1, (1,): 1},
                           db_name = 'prueba')

#gen.create_structures()

# stoichiometry=[Z1]*4+[Z2]*8+[Z3]*16
blocks = [('Mg', 4), ('Al',8),('O', 16)],
atom_num = [atomic_numbers[name] for (name, number) in self.blocks]
so = 