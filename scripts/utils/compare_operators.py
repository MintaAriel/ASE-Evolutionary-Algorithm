from ase.ga.data import DataConnection
from ea.analysis import operators_ase
from ase.io import read
from ase.visualize import view
import argparse
from pathlib import Path

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
    operators_ase.make_stats(operator=args.operator,
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
experiment = operators_ase.Operator_comparator([('C', 7), ('H', 8), ('N', 4), ('O', 2)]*8, molecular=True)
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