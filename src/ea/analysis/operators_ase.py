import os
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.data import DataConnection
from ase.ga.utilities import CellBounds, closest_distances_generator
from ase.data import atomic_numbers
from ase import Atoms
from ase.io import read
from ase.visualize import view


from gulp_rel import Gulp_relaxation
from ase.ga.soft_mutation import SoftMutation
import pandas as pd
import argparse
from pathlib import Path

class Operator_comparator():
    def __init__(self, blocks, molecular=False):
        self.block = blocks
        self.molecular = molecular
        self.atom_num = [atomic_numbers[name] for (name, number) in self.block]
        self.n_top = sum(value for element, value in self.block)
        self.slab = Atoms('', pbc=True)
        self.blmin = closest_distances_generator(self.atom_num, 0.1)
        self.cellbounds = CellBounds(
                    bounds={
                        'phi': [20, 160],
                        'chi': [20, 160],
                        'psi': [20, 160],
                        'a': [2, 40],
                        'b': [2, 40],
                        'c': [2, 40],
                    }
                )
        self.blmin_soft = closest_distances_generator(self.atom_num, 0.1)


    def heredity(self, mom, dad):
        self.slab.set_cell(mom.cell)
        # self.blmin = None
        pairing = CutAndSplicePairing(
        self.slab,
        self.n_top,
        self.blmin,
        p1=1.0,
        p2=0.0,
        minfrac=0.15,
        number_of_variable_cell_vectors=3,
        cellbounds=self.cellbounds,
        use_tags=self.molecular,)

        baby  = pairing.cross(mom, dad)
        return baby

    def softmutation_used_modes(self, mom, json_dir):
        mom.info['confid'] = 1
        sfmut = SoftMutation(self.blmin_soft, bounds=[2.0, 5.0], used_modes_file=json_dir, use_tags=self.molecular)
        baby = sfmut.mutate(mom)
        return baby

    def softmut_no_modes(self,mom):
        mom.info['confid'] = 1
        sfmut = SoftMutation(self.blmin_soft, bounds=[2.0, 5.0], used_modes_file=None, use_tags=self.molecular)
        baby = sfmut.mutate(mom)
        return baby


def save_progress(df, filename='progress_log.csv'):
    """Save/append DataFrame with timestamp to track progress"""
    # Check if file exists
    if os.path.exists(filename):
        # Append without writing header
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        # Create new file with header
        df.to_csv(filename, mode='w', header=True, index=False)


def make_stats(operator,moms, dads, folder, db, n_experiments, index):

    da = DataConnection(db)
    experiment = Operator_comparator([('Mg', 4), ('Al', 8), ('O', 16)])
    df = pd.DataFrame()
    os.mkdir(f'{folder}/{operator}')

    a_norelax = da.get_atoms(moms)
    b_norelax = da.get_atoms(dads)

    relax_parent_dir = Gulp_relaxation(f'/{folder}/parents')
    a = relax_parent_dir.use_gulp_no_add(a_norelax)[0]
    b = relax_parent_dir.use_gulp_no_add(b_norelax)[0]



    energies = []
    os.mkdir(f'{folder}/{operator}/Pair_{index + 1}')

    for j in range(n_experiments):
        c = None
        count = 0
        while c is None:
            try:
                if operator == 'softmut_modes':
                    c = experiment.softmutation_used_modes(a, f'{folder}/{operator}/Pair_{index+1}/used_modes.json')
                elif operator == 'softmut_nomodes':
                    c = experiment.softmut_no_modes(a)
                elif operator == 'heredity':
                    c = experiment.heredity(a, b)
                else:
                    print('The operator is not correct')
            except Exception as error:
                print('Error when creating offspring', error)
                count += 1

            if count == 3:
                break


        if count == 3:
            break

        relaxation = Gulp_relaxation(f'/{folder}/{operator}/Pair_{index+1}/{j+1}')
        try:
            energy = relaxation.use_gulp_no_add(c)

        except Exception as error:
            energy = [0,0]
            print('There was a mistake in GULP')

        energies.append(energy[1])
        print(f'{j + 1} la energia es {energy[1]}')



    df[f'Test_{index}'] = energies
    df.to_csv(f'{folder}/{operator}/progress_log.csv', mode='w', header=True, index=True)


# BEST_ASE = read('/home/vito/PythonProjects/ASEProject/EA/new_minimum/ase/BEST_ASE.vasp')
# relaxation = Gulp_relaxation(f'/home/vito/PythonProjects/ASEProject/EA/COMPARISON/Heredity/ASE_1gulp')
# relaxation.use_gulp_no_add(BEST_ASE)

def main():
    ap = argparse.ArgumentParser(description="Compare ase operators")
    ap.add_argument("--count", type=int, default=1, help="How many runs to create")
    ap.add_argument("--operator", type=str, default="None", help="Name of the operator to compare")
    ap.add_argument("--folder", default=".", help="Folder to store the data")
    ap.add_argument("--db", default=".", help="path to db")
    ap.add_argument("--pair", type=int, help="number from 0 to 9")
    args = ap.parse_args()

    folder_dir = Path(args.folder).resolve()
    db_dir = Path(args.db).resolve()
    moms = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    dads = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    make_stats(operator=args.operator,
                moms= moms[args.pair],
                dads = dads[args.pair],
                folder = folder_dir,
               db=db_dir,
               n_experiments=args.count,
               index=args.pair)

# if __name__ == "__main__":
#     main()

da = DataConnection('/home/vito/PythonProjects/ASEProject/Molcrys/theophylline/database/theophylline_8.db')
a = da.get_atoms(3)
b = da.get_atoms(3)
a.wrap()
b.wrap()
tags = a.get_tags()
view(a)
# view(b)
#
experiment = Operator_comparator([('C', 7), ('H', 8), ('N', 4), ('O', 2)]*8, molecular=True)
print(len(a), len(b), experiment.n_top)
print(a.get_tags())
her= None
while her == None:
    her = experiment.softmutation_used_modes(a,'/home/vito/PythonProjects/ASEProject/EA/MOLCRYS/test/soft.json')
    # her = experiment.(a)
    print('trying')

import numpy as np

print(her)

her.set_tags(tags)

single_molecule = (her.get_tags() == 1)
molecule = her.__getitem__(single_molecule)
view(her)
view(molecule)
# print(da.get_slab())