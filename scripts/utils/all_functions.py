from pathlib import Path
from ase.ga.data import DataConnection
from EA.crystal_ea.utils.mol_inspector import Molecule_inspector
script_dir = Path(__file__).parent
project_root = script_dir.parent
from ase.io import read
from pGFNFF_opti.test_gulp import GULP_test_opt

def try_test():
    theof = read('/home/vito/PythonProjects/ASEProject/Molcrys/nicotinamide/cif/77.cif')
    example_dir_data = '/home/vito/PythonProjects/ASEProject/Molcrys/nicotinamide'
    example_out = '/home/vito/PythonProjects/ASEProject/Molcrys/test/CalcFold_fixed/test'


    param_test = GULP_test_opt(mol_data_dir=example_dir_data,
                               out_dir=example_out,
                               ase_crystal=theof,
                               n_mol = 4,
                               atoms_order = False)
    param_test.gulp_lib = 'lennard'
    param_test.test_relax(bonds=True)
    bond_lenght = param_test.create_traj()
    print(bond_lenght)
    param_test.plot_bond_change(bond_lenght, [2,3])

try_test()

def try_inspector():
    inspector = Molecule_inspector(
        conections_dir='/home/vito/PythonProjects/ASEProject/EA/MOLCRYS/gen_track/conections',
        Mol_dir='/home/vito/PythonProjects/ASEProject/EA/MOLCRYS/gen_track/MOL_1',
        bond_change=0.4,
        tag=2)

    da = DataConnection(
        '/home/vito/PythonProjects/ASEProject/EA/MOLCRYS/Genesis/try/Nicotinamide_2/nicotinamide_1/nicotinamide_zahra.db')
    for i in range(50, 100):
        atom = da.get_atoms(i)
        chek_molecule = inspector.bond_inspection(atom)
        if chek_molecule == True:
            print(i)
            print(atom.info['key_value_pairs']['origin'])

        else:
            if atom.info['key_value_pairs']['origin'] == 'StrainMutation':
                print('THIS IS A GOOD STRAIN MUT', i)