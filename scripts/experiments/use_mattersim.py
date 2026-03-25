from ase.build import bulk
from pathlib import Path
import os
# from ea.io.uspex_io import best_POSCAR
from mattersim.forcefield import MatterSimCalculator
from ase.io import read, write
from ase.optimize import BFGS
from ase.io.trajectory import Trajectory
from io import StringIO
from ase.constraints import ExpCellFilter
from ase.ga.data import DataConnection
from pygulp.molecule import fix_mol_gradient
from ase.atoms import Atoms
from ase.optimize import FIRE, LBFGS
from ase.constraints import UnitCellFilter

def run_relax(atoms, calc):
    """
    Perform structural relaxation using FIRE followed by staged LBFGS.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic structure to relax
    calc : ase.calculators.calculator.Calculator
        Calculator to attach

    Returns
    -------
    dict
        Summary of the relaxation process
    """

    # Attach calculator
    atoms.calc = calc

    # Initial energy (optional but often useful)
    energy = atoms.get_potential_energy()

    # --- FIRE stage ---
    fire = FIRE(
        UnitCellFilter(atoms, hydrostatic_strain=False, constant_volume=False),
        logfile=None,
        trajectory=None,
        maxstep=0.03,
    )
    fire.run(fmax=0.10, steps=500)
    fire_steps = int(fire.nsteps)

    # --- LBFGS staged refinement ---
    lbfgs = LBFGS(
        UnitCellFilter(atoms, hydrostatic_strain=False, constant_volume=False),
        logfile=None,
        trajectory=None,
        maxstep=0.03,
        memory=40,
    )

    stages = [0.03, 0.01, 0.005, 0.002, 0.001]

    for fmax in stages:
        lbfgs.run(fmax=fmax, steps=1200)

    lbfgs_steps = int(lbfgs.nsteps)
    total_steps = fire_steps + lbfgs_steps

    return {
        "optimizer": "fire_then_staged_lbfgs",
        "initial_energy": energy,
        "final_energy": atoms.get_potential_energy(),
        "nsteps": total_steps,
        "fire_steps": fire_steps,
        "lbfgs_steps": lbfgs_steps,
        "lbfgs_stages": stages,
    }

script_dir = Path(__file__).parent
project_root = script_dir.parent



vita_container = '/home/vito/PythonProjects/ASEProject/container_gpu_2/models/tuned_mattersim_12.03.2026.pth'
traj_dir = os.path.join(project_root, 'test', 'matersim','experimental.traj')
traj_dir1 = os.path.join(project_root, 'test', 'matersim','experimental1.traj')
tunned_matersim = os.path.join(project_root, 'models', 'tuned_mattersim_12.03.2026', 'best_model')


# best_poscar = best_POSCAR('/home/vito/PythonProjects/ASEProject/EA/test/collected_theophylline/BESTgatheredPOSCARS/BESTgatheredPOSCARS_test_26')

# best = best_poscar.get_best()
# atoms1 = read(StringIO(best['EA3372']), format='vasp')
calc = MatterSimCalculator.from_checkpoint(
    load_path=vita_container,
    device="cuda"
)

traj = Trajectory('/home/vito/PythonProjects/ASEProject/EA/test/matersim/33_bc.traj')
da = DataConnection('/home/vito/uspex_matlab/theo_pyxtal/test_1/theophilline.db')

write('/home/vito/PythonProjects/ASEProject/EA/test/matersim/best.cif',traj[-1])


# best  = read('/home/vito/Downloads/862238.cif')
best = read('/home/vito/PythonProjects/ASEProject/container_gpu_2/structures/1039798.cif')
# atoms1 = read('/home/vito/Downloads/mybest.cif')

result = run_relax(best, calc)
print(result)
# atoms = da.get_atoms(8)
# print(atoms.todict())

connection_dir = '/home/vito/PythonProjects/ASEProject/EA/data/theophylline/connections'


# optimizer = fix_mol_gradient.GradientDescentGULP(atoms, work_dir=os.path.join(project_root, 'test', 'matersim'), connections=connection_dir, library='reaxff')

# optimizer.asu.cell[1] *= 2
# optimizer.asu.cell[2] *= 0.3
#
# new  = optimizer.run(steps =50,  traj=True)
# best = optimizer.best_structure
#
#
# best.calc = calc
# atoms1.calc = calc
#
# mask = [1, 1, 1, 0, 0, 0]   # allow only normal strains
# ecf = ExpCellFilter(best, mask)
# opt = BFGS(ecf, trajectory=traj_dir)
# opt.run(fmax=0.001)
#
# # ecf1 = ExpCellFilter(atoms1)
# # opt1 = BFGS(ecf1, trajectory=traj_dir1)
# # opt1.run(fmax=0.02)
#
#
# energy = best.get_potential_energy()
#
# print(energy)


