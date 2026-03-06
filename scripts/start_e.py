import os
from ase_creator import mol2ase
from read_input import parse_input_file
from structure_generator import first_gen_mol
import argparse
import shutil

def clean_workdir(workdir, keep_files=("MOL_1", "INPUT.txt")):
    """
    Deletes all files and directories inside workdir
    except those listed in keep_files.
    """

    for item in os.listdir(workdir):
        if item in keep_files:
            continue

        path = os.path.join(workdir, item)

        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

def main():

    parser = argparse.ArgumentParser(
        description="Initial molecular population generator"
    )

    parser.add_argument(
        "workdir",
        nargs="?",
        default=".",
        help="Working directory containing INPUT.txt and MOL_1"
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete all files in working directory except MOL_1 and INPUT.txt"
    )

    args = parser.parse_args()

    # Convert to absolute path for safety
    test_dir = os.path.abspath(args.workdir)

    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Directory does not exist: {test_dir}")

    input_path = os.path.join(test_dir, "INPUT.txt")
    mol_path = os.path.join(test_dir, "MOL_1")
    db_path = os.path.join(test_dir, "theophilline.db")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing INPUT.txt in {test_dir}")

    if not os.path.exists(mol_path):
        raise FileNotFoundError(f"Missing MOL_1 in {test_dir}")


    # Perform cleaning if flag is used
    if args.clean:
        print(f"Cleaning directory: {test_dir}")
        clean_workdir(test_dir)

    # Read INPUT.txt from current folder
    params = parse_input_file(os.path.join(test_dir, 'INPUT.txt'))


    ase_mol = mol2ase(MOL_path=mol_path)
    molecule = ase_mol.read()

    n = params['numSpecies']

    gen = first_gen_mol(blocks = [(molecule, n)],
                               population=params['initialPopSize'],
                               volume = 300* params['volume_per_molecule'],
                               splits = params['splits'],
                               db_path = db_path,
                               symmetry = True)
    gen.create_structures()

if __name__ == "__main__":
    main()

'''
atomic crystal
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
                           db_name = 'MoB_2.db',
                           symmetry=True,
                           varcomp=[8,18],
                            build_blocks=[[1,0],[0,1]])

'''
