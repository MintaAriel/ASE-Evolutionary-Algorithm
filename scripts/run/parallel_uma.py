from ea.simulators.uma_relax import UmaRelax
from ea.parallel.create_batch import batch_calculator_uma
from ea.parallel.FIRE_parallel import ParallelFIRE
import numpy as np
from ase.optimize import FIRE
from ase.io import read
from ase.filters import FrechetCellFilter
import warnings
warnings.filterwarnings("ignore", message=r"logm result may be inaccurate.*")


test_traj = '/home/vito/uspex_matlab/theo_pyxtal/2THP/test_1 (Copy)/output.traj'
images = read(test_traj, index=':')  # Returns a list of Atoms objects

uma = UmaRelax(model_name='uma-s-1p1', device='cuda')
calc = uma.get_calc()


def make_uma_evaluator(calculator):
    """Wrap batch_calculator_uma into the (atoms_list) -> (E, F, S) signature
    that ParallelFIRE expects."""
    def _eval(atoms_list):
        return batch_calculator_uma(atoms_list, calculator)
    return _eval


def run_full_optimization(batch_filtered, calc, out_dir, batch_size: int | None = None):
    if batch_size is None:
        batch_size = len(batch_filtered)
    batch = [c.copy() for c in batch_filtered[:batch_size]]

    evaluator = make_uma_evaluator(calc)

    print("\n=== initial batched eval ===")

    # E0, F0, V0 = evaluator(batch)
    # for i in range(len(batch)):
    #     fmax0 = np.linalg.norm(np.asarray(F0[i]).reshape(-1, 3), axis=1).max()
    #     vol = batch[i].get_volume()
    #     stress_voigt = np.asarray(V0[i]).reshape(6)
    #     print(f"  struct {i}: E = {float(np.asarray(E0[i]).ravel()[0]):.4f}  "
    #           f"fmax = {fmax0:.4f}  "
    #           f"tr(stress) = {stress_voigt[:3].sum():.4f} eV/A^3")
    # print(E0)
    # indices = [i for i, x in enumerate(E0) if x > -545]
    # print(indices)

    print("\n=== running ParallelFIRE (positions + cell) ===")

    opt = ParallelFIRE(batch, batch_evaluator=evaluator,
                       fmax=0.1, max_steps=500, maxstep=0.03)
    opt.run()
    batch = opt.get_atoms()

    print("\n=== final energies ===")
    for i, s in enumerate(opt.states):
        tag = "CONVERGED" if s.converged else "not converged"
        print(f"  struct {i}: E = {s.energy:.4f}  "
              f"fmax = {s.fmax_current:.4f}  "
              f"vol = {s.atoms.get_volume():.2f}  [{tag}]")


if __name__ == '__main__':
    # for v in images[:10]:
    #     v.calc = calc
    #     fire = FIRE(
    #         FrechetCellFilter(v),
    #         maxstep=0.03,
    #     )
    #     fire.run(fmax=0.1, steps=500)
    #     print('DONE')

    run_full_optimization(images, calc, out_dir=None, batch_size=10)
