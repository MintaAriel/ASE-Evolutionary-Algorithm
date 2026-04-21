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

cfg = load_config()

MODELS = {
    'base_deepmd': 'dpa3_12.03.2026.pth',
    'deepmd_d3': 'dpa3-d3_torch.pth',
    'deepmd_d4': 'dpa3-d4.pth',
    'deepmd_d3_abs': 'dpa3-d3_abs_torch.pth',
    'deepmd_d3_mbj': 'dpa3-d3-cpu_mbj.pth',
    'deepmd_d3_mbj_abs': 'dpa3-d3-cpu_mbj_abs.pth',
}

def resolve_model(models_dir: Path, key: str) -> tuple[Path, str]:
    path = Path(models_dir) / Path(MODELS[key])
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")
    return path, key

def create_calc(model_name:str = 'deepmd_d3'):
    path,_ = resolve_model(cfg['deepmd']['models_path'], model_name)
    calculator = DP(model=path, device='gpu')
    return calculator


def create_batch_from_seeds():
    pths = cfg['paths']
    project_root = Path(__file__).resolve().parents[2]
    results_dir = Path(project_root / pths['results_dir']).resolve()
    test_to_try = [41]

    # Output directory for this experiment
    output_dir = results_dir / 'THP' / 'relax_first_generation'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored in: {output_dir}")

    tests = [  # 'collected_theophylline_0',
        # 'collected_theophylline_1',
        'collected_theophylline_3']
    # 'theophylline_uspex']
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

        return  batch

def crete_batch_from_traj():
    atoms_list = read('output.traj', index=':')
    return atoms_list


def clean_batch(idx:list, batch):
    for i in sorted(idx, reverse=True):
        del batch[i]
    print(batch)
    return batch


def point_calculation(batch, calc, energy_threshold:float = 0):
    print("\n=== initial batched eval ===")
    coords0, cells0, types0 = build_batch_deepmd(batch, calc.type_dict)
    E0, F0, V0 = calc.dp.eval(coords0, cells0, types0)[:3]
    for i in range(len(batch)):
        fmax0 = np.linalg.norm(np.asarray(F0[i]).reshape(-1, 3), axis=1).max()
        vol = batch[i].get_volume()
        stress = -np.asarray(V0[i]).reshape(3, 3) / vol
        print(f"  struct {i}: E = {float(np.asarray(E0[i]).ravel()[0]):.4f}  "
              f"fmax = {fmax0:.4f}  "
              f"tr(stress) = {stress.trace():.4f} eV/A^3")
    print(E0)
    indices = [i for i, x in enumerate(E0) if x > energy_threshold]
    print(indices)


    return indices


def run_full_optimization(batch_filtered, calc, out_dir, batch_size: int | None = None):
    if batch_size is None:
        batch_size = len(batch_filtered)
    batch = [c.copy() for c in batch_filtered[:batch_size]]

    print("\n=== initial batched eval ===")
    coords0, cells0, types0 = build_batch_deepmd(batch, calc.type_dict)
    E0, F0, V0 = calc.dp.eval(coords0, cells0, types0)[:3]
    for i in range(len(batch)):
        fmax0 = np.linalg.norm(np.asarray(F0[i]).reshape(-1, 3), axis=1).max()
        vol = batch[i].get_volume()
        stress = -np.asarray(V0[i]).reshape(3, 3) / vol
        print(f"  struct {i}: E = {float(np.asarray(E0[i]).ravel()[0]):.4f}  "
              f"fmax = {fmax0:.4f}  "
              f"tr(stress) = {stress.trace():.4f} eV/A^3")
    print(E0)
    indices = [i for i, x in enumerate(E0) if x > -545]
    print(indices)

    print("\n=== running ParallelFIRE (positions + cell) on 4 structures ===")
    print('batch before', batch)

    opt = ParallelFIRE(batch, calc, fmax=0.1, max_steps=500, maxstep=0.03)
    opt.run()
    batch = opt.get_atoms()

    print("\n=== final energies ===")
    for i, s in enumerate(opt.states):
        tag = "CONVERGED" if s.converged else "not converged"
        print(f"  struct {i}: E = {s.energy:.4f}  "
              f"fmax = {s.fmax_current:.4f}  "
              f"vol = {s.atoms.get_volume():.2f}  [{tag}]")

    print("\n=== running ParallelLBFGS (positions + cell) on 4 structures ===")
    opt1 = ParallelLBFGS(batch, calc, fmax=0.03, max_steps=1200, maxstep=0.03, memory=40)
    opt1.run(out_dir=out_dir)
    batch = opt1.get_atoms()

    print('fmax=0.01')
    opt2 = ParallelLBFGS(batch, calc, fmax=0.01, max_steps=1200, maxstep=0.03, memory=40)
    opt2.run(out_dir=out_dir)
    batch = opt2.get_atoms()

    print('fmax=0.005')
    opt3 = ParallelLBFGS(batch, calc, fmax=0.005, max_steps=1200, maxstep=0.03, memory=40)
    opt3.run(out_dir=out_dir)
    batch = opt3.get_atoms()

    print('fmax=0.002')
    opt4 = ParallelLBFGS(batch, calc, fmax=0.002, max_steps=1200, maxstep=0.03, memory=40)
    opt4.run(out_dir=out_dir)
    batch = opt4.get_atoms()

    print('fmax=0.001')
    opt5 = ParallelLBFGS(batch, calc, fmax=0.001, max_steps=1200, maxstep=0.03, memory=40)
    opt5.run(out_dir=out_dir)
    batch = opt5.get_atoms()

    print("\n=== final energies ===")
    for i, s in enumerate(opt1.states):
        tag = "CONVERGED" if s.converged else "not converged"
        print(f"  struct {i}: E = {s.energy:.4f}  "
              f"fmax = {s.fmax_current:.4f}  "
              f"vol = {s.atoms.get_volume():.2f}  [{tag}]")


# --- quick test with 4 structures ---------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Load a trajectory, score all frames with point_calculation, drop "
            "frames above an energy threshold, then run run_full_optimization "
            "on the survivors."
        )
    )
    parser.add_argument(
        "traj",
        type=Path,
        help="Path to the .traj/.trajectory file (all frames are read).",
    )
    parser.add_argument(
        "out_dir",
        type=Path,
        help="Directory where run_full_optimization writes its outputs.",
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=0.0,
        help="Frames with E > threshold are filtered out before optimization.",
    )
    parser.add_argument(
        "--model",
        default="deepmd_d3",
        help=f"Model key from MODELS dict. One of: {', '.join(MODELS)}.",
    )
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Stop after point_calculation + clean_batch; skip run_full_optimization.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1,
        help="number of crystals in the batch (filtered) to optimize",
    )
    args = parser.parse_args()

    traj_path = args.traj.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    batch_size = args.size

    if not traj_path.is_file():
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading trajectory: {traj_path}")
    batch = read(str(traj_path), index=":")
    if not isinstance(batch, list):
        batch = [batch]
    print(f"Loaded {len(batch)} structures")

    calc = create_calc(args.model)

    indices_to_drop = point_calculation(
        batch, calc, energy_threshold=args.energy_threshold
    )
    print(f"Dropping {len(indices_to_drop)} structures with E > "
          f"{args.energy_threshold}: {indices_to_drop}")

    batch_filtered = clean_batch(indices_to_drop, batch)
    print(f"Batch size after filtering: {len(batch_filtered)}")

    if args.skip_optimization:
        print("--skip-optimization set; stopping before run_full_optimization.")
    elif not batch_filtered:
        print("No structures left after filtering; nothing to optimize.")
    else:
        print(f"Output directory: {out_dir}")
        run_full_optimization(batch_filtered, calc, out_dir, batch_size)
