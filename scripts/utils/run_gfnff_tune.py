"""Runner for the mcGFN-FF Bayesian optimization on the DeepmdTrain dataset.

Uses ea.simulators.fine_tune_gfnff.GfnffTuner against the VASP reference
trajectory, then rebuilds and plots the best traj via CompareModel.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ea.simulators.fine_tune_gfnff import (
    GfnffTuner,
    compute_loss,
    plot_comparison,
    recalculate_batch_gulp,
    DEFAULT_INITIAL_PARAMS,
)


REF_TRAJ = Path('/home/vito/PythonProjects/ASEProject/DeepmdTrain/allpbebj_vasp.traj')
WORK_ROOT = Path('/home/vito/PythonProjects/ASEProject/DeepmdTrain/gfnff_tune')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n-trials', type=int, default=40)
    p.add_argument('--study-name', default='gfnff_tune')
    p.add_argument('--ref', type=Path, default=REF_TRAJ)
    p.add_argument('--work-root', type=Path, default=WORK_ROOT)
    p.add_argument('--delta', type=float, default=0.3)
    p.add_argument('--n-jobs', type=int, default=1,
                   help='parallel GULP workers per trial')
    p.add_argument('--cpu-pins', type=str, default=None,
                   help='comma-separated CPU ids to pin workers to '
                        '(length must equal --n-jobs), e.g. "0,1,2,3,4"')
    p.add_argument('--smoke', action='store_true',
                   help='single evaluation at the initial parameters')
    return p.parse_args()


def _parse_pins(s, n_jobs):
    if s is None:
        return None
    pins = [int(x) for x in s.split(',') if x.strip()]
    if len(pins) != n_jobs:
        raise SystemExit(f'--cpu-pins has {len(pins)} ids but --n-jobs={n_jobs}')
    return pins


def main():
    args = parse_args()
    pins = _parse_pins(args.cpu_pins, args.n_jobs)

    if args.smoke:
        smoke_dir = args.work_root / 'smoke'
        traj_path = smoke_dir / 'gfnff.traj'
        n = recalculate_batch_gulp(
            DEFAULT_INITIAL_PARAMS,
            args.ref,
            traj_path,
            smoke_dir / 'gulp',
            n_jobs=args.n_jobs,
            cpu_pins=pins,
        )
        print(f'smoke wrote {n} frames')
        loss, parts = compute_loss(args.ref, traj_path)
        print(f'smoke loss={loss:.4f} parts={parts}')
        return

    tuner = GfnffTuner(
        ref_path=args.ref,
        work_root=args.work_root,
        delta=args.delta,
        study_name=args.study_name,
        n_jobs=args.n_jobs,
        cpu_pins=pins,
    )
    study = tuner.run(n_trials=args.n_trials)
    best_traj, _ = tuner.rebuild_best(study)
    plot_comparison(args.ref, best_traj,
                    save_dir=best_traj.parent, show=False)


if __name__ == '__main__':
    main()
