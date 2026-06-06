"""Bayesian optimization of mcGFN-FF gfnff_scale parameters.

Re-runs single-point GULP/gfnff on a reference trajectory with a candidate
parameter vector, then minimises a weighted RMSE loss against the reference:

    Loss = w_e * RMSE(E/atom) + w_f * RMSE(F) + w_v * RMSE(virial/atom)
"""
from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

import numpy as np
import optuna
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import Trajectory, read

from ea.analysis.compare_model import CompareModel
from pygulp.io.read_gulp import read_results
from pygulp.relaxation.relax import Gulp_relaxation_noadd


DEFAULT_INITIAL_PARAMS = (0.80, 1.343, 0.727, 1.0, 2.859)
DEFAULT_WEIGHTS = {'energy': 0, 'forces': 0.6, 'virial': 0.2}


def gulp_singlepoint(atom, gfnff_scale, calc_dir: Path):
    scale_str = ' '.join(f'{v:.6f}' for v in gfnff_scale)
    gulp_input = (
        "gradient conp conse qok c6 conp prop gfnff gwolf noauto\n"
        f"gfnff_scale {scale_str}\n"
        "maths mrrr\n"
        "pressure 0 GPa"
    )
    options = "output movie cif out1.cif\nmaxcycle 300\ngtol 0.00001"

    relax = Gulp_relaxation_noadd(
        path=str(calc_dir),
        library=None,
        gulp_keywords=gulp_input,
        gulp_options=options,
    )
    new = relax.use_gulp(atom)
    results = read_results(str(calc_dir / 'CalcFold' / 'ginput1.got'))
    return new, results


def _eval_frame(atom, gfnff_scale, calc_dir: Path):
    """Single-frame gfnff SP -> SinglePointCalculator-bound copy of atom."""
    _, results = gulp_singlepoint(atom.copy(), gfnff_scale, calc_dir)
    energy = results['energy'][0]
    grad_frac = results['gradient']
    A = atom.cell.array
    forces = -(grad_frac @ np.linalg.inv(A))
    vt = results['strain']
    virial = np.array([vt[0, 0], vt[1, 1], vt[2, 2],
                       vt[1, 2], vt[0, 2], vt[0, 1]])
    stress = -virial / atom.get_volume()

    a = atom.copy()
    a.calc = SinglePointCalculator(a, energy=energy, forces=forces, stress=stress)
    return a


def recalculate_batch_gulp(gfnff_scale, ref_path: Path, out_path: Path,
                           calc_dir: Path,
                           n_jobs: int = 1,
                           cpu_pins: list[int] | None = None):
    """Run gfnff single-points on every frame of ref_path with the given scale.

    Writes an ase Trajectory at out_path with energy / forces / stress attached.
    Returns the number of frames successfully written.

    Parameters
    ----------
    n_jobs:
        Number of parallel GULP workers. Each worker uses its own subdirectory
        of calc_dir (worker_0/, worker_1/, ...) so they do not interfere.
        OMP_NUM_THREADS is forced to 1 so each gulp process stays single-threaded.
    cpu_pins:
        Optional list of CPU core ids to pin workers to (length must match
        n_jobs). Worker w runs only on cpu_pins[w]. Pinning is set on the
        worker thread; the spawned gulp subprocess inherits it.
    """
    ref_path = Path(ref_path)
    out_path = Path(out_path)
    calc_dir = Path(calc_dir)

    if cpu_pins is not None and len(cpu_pins) != n_jobs:
        raise ValueError(
            f'cpu_pins length ({len(cpu_pins)}) must equal n_jobs ({n_jobs})'
        )

    batch = read(str(ref_path), index=':')
    calc_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # keep each gulp subprocess on a single core
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    n = len(batch)
    results: list = [None] * n

    if n_jobs <= 1:
        for i, atom in enumerate(batch):
            try:
                results[i] = _eval_frame(atom, gfnff_scale, calc_dir)
            except Exception as e:
                print(f'  frame {i} failed: {e}')
    else:
        slot_q: Queue = Queue()
        for w in range(n_jobs):
            wdir = calc_dir / f'worker_{w}'
            wdir.mkdir(parents=True, exist_ok=True)
            pin = cpu_pins[w] if cpu_pins is not None else None
            slot_q.put((w, wdir, pin))

        def task(idx_atom):
            idx, atom = idx_atom
            wid, wdir, pin = slot_q.get()
            try:
                if pin is not None:
                    try:
                        os.sched_setaffinity(0, {pin})
                    except (OSError, AttributeError) as e:
                        print(f'  worker {wid}: pin to cpu {pin} failed: {e}')
                try:
                    return idx, _eval_frame(atom, gfnff_scale, wdir)
                except Exception as e:
                    print(f'  frame {idx} (worker {wid}) failed: {e}')
                    return idx, None
            finally:
                slot_q.put((wid, wdir, pin))

        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            for idx, sp in ex.map(task, list(enumerate(batch))):
                results[idx] = sp

    written = 0
    with Trajectory(str(out_path), mode='w') as traj_out:
        for sp in results:
            if sp is not None:
                traj_out.write(sp)
                written += 1
    return written


def compute_loss(ref_path: Path, model_path: Path,
                 weights: dict | None = None):
    """Weighted RMSE per atom, matching the CompareModel collection."""
    weights = weights or DEFAULT_WEIGHTS
    cmp = CompareModel(str(ref_path), str(model_path))
    e1, e2, f1, f2, v1, v2 = cmp._collect()
    rmse_e = float(np.sqrt(np.mean((e1 - e2) ** 2)))
    rmse_f = float(np.sqrt(np.mean((f1 - f2) ** 2)))
    rmse_v = float(np.sqrt(np.mean((v1 - v2) ** 2)))
    loss = (weights['energy'] * rmse_e
            + weights['forces'] * rmse_f
            + weights['virial'] * rmse_v)
    return loss, {'rmse_e': rmse_e, 'rmse_f': rmse_f, 'rmse_v': rmse_v}


class GfnffTuner:
    """Bayesian-optimization driver for mcGFN-FF gfnff_scale parameters."""

    def __init__(self, ref_path: Path, work_root: Path,
                 initial_params=DEFAULT_INITIAL_PARAMS,
                 delta: float = 0.3,
                 weights: dict | None = None,
                 study_name: str = 'gfnff_tune',
                 sampler_seed: int = 42,
                 n_jobs: int = 1,
                 cpu_pins: list[int] | None = None):
        self.ref_path = Path(ref_path)
        self.work_root = Path(work_root)
        self.initial_params = np.asarray(initial_params, dtype=float)
        self.delta = float(delta)
        self.weights = weights or DEFAULT_WEIGHTS
        self.study_name = study_name
        self.sampler_seed = sampler_seed
        self.n_jobs = int(n_jobs)
        self.cpu_pins = list(cpu_pins) if cpu_pins is not None else None

        self.work_root.mkdir(parents=True, exist_ok=True)

    @property
    def low(self):
        return self.initial_params * (1.0 - self.delta)

    @property
    def high(self):
        return self.initial_params * (1.0 + self.delta)

    def _objective(self, trial: optuna.Trial):
        theta = np.array([
            trial.suggest_float(f'p{i+1}', self.low[i], self.high[i])
            for i in range(len(self.initial_params))
        ])
        trial_dir = self.work_root / f'trial_{trial.number:04d}'
        traj_path = trial_dir / 'gfnff.traj'
        calc_dir = trial_dir / 'gulp'

        try:
            n = recalculate_batch_gulp(theta, self.ref_path, traj_path, calc_dir,
                                       n_jobs=self.n_jobs, cpu_pins=self.cpu_pins)
            ref_n = len(read(str(self.ref_path), index=':'))
            if n != ref_n:
                print(f'  trial {trial.number}: only {n}/{ref_n} frames succeeded')
                return float('inf')

            loss, parts = compute_loss(self.ref_path, traj_path, self.weights)
            trial.set_user_attr('rmse_e', parts['rmse_e'])
            trial.set_user_attr('rmse_f', parts['rmse_f'])
            trial.set_user_attr('rmse_v', parts['rmse_v'])
            trial.set_user_attr('params', theta.tolist())
            print(f'  trial {trial.number}: theta={theta.tolist()}  '
                  f"E={parts['rmse_e']:.4f}  F={parts['rmse_f']:.4f}  "
                  f"V={parts['rmse_v']:.4f}  loss={loss:.4f}")
            return loss
        finally:
            shutil.rmtree(trial_dir, ignore_errors=True)

    def run(self, n_trials: int):
        storage = f'sqlite:////{(self.work_root / f"{self.study_name}.db").resolve()}'
        print(f'storage: {storage}')

        sampler = optuna.samplers.TPESampler(seed=self.sampler_seed)
        study = optuna.create_study(
            study_name=self.study_name,
            storage=storage,
            direction='minimize',
            load_if_exists=True,
            sampler=sampler,
        )
        study.enqueue_trial(
            {f'p{i+1}': float(v) for i, v in enumerate(self.initial_params)},
            skip_if_exists=True,
        )
        study.optimize(self._objective, n_trials=n_trials)
        print('best params:', study.best_params)
        print('best loss :', study.best_value)
        return study

    def best_params_array(self, study: optuna.Study):
        return np.array([study.best_params[f'p{i+1}']
                         for i in range(len(self.initial_params))])

    def rebuild_best(self, study: optuna.Study, out_dir: Path | None = None):
        out_dir = Path(out_dir) if out_dir else self.work_root / 'best'
        out_dir.mkdir(parents=True, exist_ok=True)
        best = self.best_params_array(study)
        traj = out_dir / 'gfnff.traj'
        print(f'rebuilding best traj at {traj} with params={best.tolist()}')
        recalculate_batch_gulp(best, self.ref_path, traj, out_dir / 'gulp',
                               n_jobs=self.n_jobs, cpu_pins=self.cpu_pins)
        return traj, best


def plot_comparison(ref_path: Path, model_traj: Path,
                    save_dir: Path | None = None, show: bool = False):
    cmp = CompareModel(str(ref_path), str(model_traj))
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        cmp.plot(save_path=str(save_dir / 'parity.png'), show=show)
        cmp.plot_forces_xyz(save_path=str(save_dir / 'forces_xyz.png'), show=show)
        cmp.plot_virial_diag(save_path=str(save_dir / 'virial_diag.png'), show=show)
    else:
        cmp.plot(show=show)
        cmp.plot_forces_xyz(show=show)
        cmp.plot_virial_diag(show=show)
