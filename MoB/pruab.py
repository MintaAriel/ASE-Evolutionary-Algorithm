from ase.ga.data import DataConnection
from ase.ga.utilities import CellBounds, closest_distances_generator
import numpy as np

class Genetic_algorith():
    def __init__(self, database ):
        self.db = database
        self.da = DataConnection(self.db)
        self.slab = self.da.get_slab()
        self.atoms_to_opt = list(set(self.da.get_atom_numbers_to_optimize()))
        self.n_top = len(self.atoms_to_opt)
        self.blmin = closest_distances_generator(self.atoms_to_opt, 0.5)

# s = Genetic_algorith('/home/vito/PythonProjects/ASEProject/EA/Programms/GA1_40.db')

# print(s.atoms_to_opt)
# print(s.blmin)

blocks = [('Mo', 1), ('B',1)]
build_blocks=[[1,0], [0,1]]

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


element_name = [t[0] for t in blocks]

for i in range(12):
    f = random_block(8,18, blocks, build_blocks)
    element_name = [t[0] for t in blocks]
    new_block = list(zip(element_name,f))
    print(new_block)

list = [list(zip(element_name,random_block(8,18, blocks, build_blocks))) for _ in range(20)]
print(list)