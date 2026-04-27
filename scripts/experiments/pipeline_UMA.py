from ase.io import read
from pathlib import Path
import os
from ase.db import connect
from ase.ga.data import DataConnection
from ase.ga.data import PrepareDB
from ase import Atoms



PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_DIR / 'test' / 'ML-train'
TRAIN_DATA_DIRS = PROJECT_DIR / 'data' / 'ML-model' / 'Data'




def create_db(out_dir, name):
    db_path = Path(out_dir) / f"{name}.db"
    slab = Atoms('', pbc=True)
    da = PrepareDB(db_file_name=db_path, simulation_cell=slab, stoichiometry='CHON')
    poscars = os.listdir(TRAIN_DATA_DIRS)
    for atom in poscars:
        atom_dir = TRAIN_DATA_DIRS / atom
        crystal = read(atom_dir, format='vasp')
        crystal.info['key_value_pairs'] = {}
        crystal.info['data'] = {}
        print(crystal)
        da.add_unrelaxed_candidate(crystal, description=f'name:{atom}')

    print(TRAIN_DATA_DIRS)
    print(poscars)

    return DataConnection(db_path)

# da = create_db(OUT_DIR, 'misha')

def create_batches(db_dir, max_atoms):
    '''

    :param db_dir: ASE DB Sqlite dir
    :param max_atoms:  max amount of atoms the gpu can handle (graph size)
    :return: a np list with the groups of indices of atoms that fits in the batch
    '''
    da = DataConnection(db_dir)
    all = da.get_all_unrelaxed_candidates()
    atoms_num =
    print(all)




create_batches('/home/vito/PythonProjects/ASEProject/EA/test/ML-train/misha.db', 3)