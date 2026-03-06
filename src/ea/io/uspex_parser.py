
from ase.io import read
import os
from ase.visualize import view



def best_poscar():
    atom = read(os.path.expanduser("~/uspex-python/results1/best_structure_POSCAR"))

    # view(atom)

    data = {}
    with open(os.path.expanduser('~/uspex-python/results1/gatheredPOSCARS_unrelaxed'), 'r') as file:
        content = file.read()
        lines_list = content.splitlines()
        for index, v in enumerate(lines_list):
            item = 0
            if 'number' in v:
                print(lines_list[index][7:])
                print(index)
                item = 1

def join_poscar():
    path = '/home/vito/PythonProjects/ASEProject/EA/new_minimum/ALL_BEST'
    seeds_dir = os.listdir(path)
    all_seeds = None
    for dir in seeds_dir:
        seeds_dir = os.path.join(path, dir)
        best = read(seeds_dir, format="vasp")
        all_seeds += best

    with open("ALL_BEST.vasp", "w") as f:
        f.write(all_seeds)