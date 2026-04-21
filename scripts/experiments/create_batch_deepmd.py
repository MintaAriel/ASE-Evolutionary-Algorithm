from ase.io import read
from ase.visualize import  view
from deepmd.calculator import DP
from pathlib import Path
from io import StringIO
from ea.parallel.create_batch import build_batch_deepmd
from ea.parallel.FIRE_parallel import ParallelFIRE
from ea.parallel.LBFGS_parallel import ParallelLBFGS
import numpy as np
from ea.io.uspex_io import get_structure_from_id
from ea.utils.config import load_config
import warnings
warnings.filterwarnings("ignore", message=r"logm result may be inaccurate.*")


# uspex_test = '/home/vito/uspex_matlab/theo_uspex/test_parallel'
#
# new_batch = [read(Path(uspex_test, f'CalcFold{i}', 'geom.in'), format='vasp') for i in range(1,39)]
#
# #Shift atoms so they dont intefere
# # create_batch(batch)
#
# print(new_batch)
#
# for i, c in enumerate(new_batch):
#     print(f"Structure {i}:")
#     print("  Natoms:", len(c))
#     print("  Positions shape:", c.get_positions().shape)
#
#
calc = DP(model='/home/vito/PythonProjects/ASEProject/container_gpu_2/models/dpa3-d3_torch.pth', device='gpu')

cfg = load_config()
pths = cfg['paths']
project_root = Path(__file__).resolve().parents[2]
results_dir = Path(project_root / pths['results_dir']).resolve()

# Output directory for this experiment
output_dir = results_dir / 'THP' / 'relax_first_generation'
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Results will be stored in: {output_dir}")

tests = [#'collected_theophylline_0',
         #'collected_theophylline_1',
         'collected_theophylline_3']
         #'theophylline_uspex']

test_to_try = [39]

batch = []
for name in tests:
    for number in test_to_try:
        poscar_dir = (results_dir / 'THP' / 'tests' / name /
                      'gatheredPOSCARS_unrelaxed' / f'gatheredPOSCARS_unrelaxed_test_{number}'
                      )
        print(f"\n--- {name} / test_{number} ---")
        print(f"POSCAR dir: {poscar_dir}")

        if not poscar_dir.exists():
            print(f"  Skipping, path does not exist: {poscar_dir}")
            continue

        poscar = get_structure_from_id(poscar_dir=poscar_dir,
                                       id_structures=[i for i in range(1, 41)])


        # Per-test output directory
        test_output_dir = output_dir / name / f'test_{number}'

        for k, v in poscar.items():
            crystal = read(StringIO(v), format='vasp')
            batch.append(crystal)

idxs = [4, 10, 15, 16, 19]

for i in sorted([3, 9, 14, 15, 18], reverse=True):
    del batch[i]
print(batch)


# --- quick test with 4 structures ---------------------------------------
if __name__ == "__main__":
    test_batch = [c.copy() for c in batch]

    print("\n=== initial batched eval ===")
    coords0, cells0, types0 = build_batch_deepmd(test_batch, calc.type_dict)
    E0, F0, V0 = calc.dp.eval(coords0, cells0, types0)[:3]
    for i in range(len(test_batch)):
        fmax0 = np.linalg.norm(np.asarray(F0[i]).reshape(-1, 3), axis=1).max()
        vol = test_batch[i].get_volume()
        stress = -np.asarray(V0[i]).reshape(3, 3) / vol
        print(f"  struct {i}: E = {float(np.asarray(E0[i]).ravel()[0]):.4f}  "
              f"fmax = {fmax0:.4f}  "
              f"tr(stress) = {stress.trace():.4f} eV/A^3")

    print("\n=== running ParallelFIRE (positions + cell) on 4 structures ===")
    print('batch before',test_batch)


    opt = ParallelFIRE(test_batch, calc, fmax=0.1, max_steps=500, maxstep=0.03)
    opt.run()
    test_batch = opt.get_atoms()


    print("\n=== final energies ===")
    for i, s in enumerate(opt.states):
        tag = "CONVERGED" if s.converged else "not converged"
        print(f"  struct {i}: E = {s.energy:.4f}  "
              f"fmax = {s.fmax_current:.4f}  "
              f"vol = {s.atoms.get_volume():.2f}  [{tag}]")


    print("\n=== running ParallelLBFGS (positions + cell) on 4 structures ===")
    opt1 = ParallelLBFGS(test_batch, calc, fmax=0.03, max_steps=1200, maxstep=0.03, memory=40)
    opt1.run(out_dir='/home/vito/PythonProjects/ASEProject/EA/test/nvidia parallel')
    test_batch = opt1.get_atoms()

    print('fmax=0.01')
    opt2 = ParallelLBFGS(test_batch, calc, fmax=0.01, max_steps=1200, maxstep=0.03, memory=40)
    opt2.run(out_dir='/home/vito/PythonProjects/ASEProject/EA/test/nvidia parallel')
    test_batch = opt2.get_atoms()

    print('fmax=0.005')
    opt3 = ParallelLBFGS(test_batch, calc, fmax=0.005, max_steps=1200, maxstep=0.03, memory=40)
    opt3.run(out_dir='/home/vito/PythonProjects/ASEProject/EA/test/nvidia parallel')
    test_batch = opt3.get_atoms()

    print('fmax=0.002')
    opt4 = ParallelLBFGS(test_batch, calc, fmax=0.002, max_steps=1200, maxstep=0.03, memory=40)
    opt4.run(out_dir='/home/vito/PythonProjects/ASEProject/EA/test/nvidia parallel')
    test_batch = opt4.get_atoms()

    print('fmax=0.001')
    opt5 = ParallelLBFGS(test_batch, calc, fmax=0.001, max_steps=1200, maxstep=0.03, memory=40)
    opt5.run(out_dir='/home/vito/PythonProjects/ASEProject/EA/test/nvidia parallel')
    test_batch = opt5.get_atoms()



    print("\n=== final energies ===")
    for i, s in enumerate(opt1.states):
        tag = "CONVERGED" if s.converged else "not converged"
        print(f"  struct {i}: E = {s.energy:.4f}  "
              f"fmax = {s.fmax_current:.4f}  "
              f"vol = {s.atoms.get_volume():.2f}  [{tag}]")
