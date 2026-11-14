"""Racemic ibuprofen (Z = 4: 2 R + 2 S) seeds + ReaxFF mutation operators, optionally in parallel.

1. Generate N random pyxtal crystals. Each one picks a random (R)-minimum and a random
   (S)-minimum from the 9 lowest PM7 minima (mol_R/min_XX_R.mol, mol_S/min_YY_S.mol)
   and places 2 copies of each.
2. Store them in a theophilline.db (the db name mutation.py expects).
3. Run the chosen operators (mutate_reaxff, mutate_reaxff_small, mutate_reaxff_small2).

With --workers W the N structures are split into W chunks; each chunk runs in its own
process (pinned to one CPU of --cpus, GULP inherits the pinning) and its own work dir
<out>/chunks/chunk_XX, so CalcFold / all_relaxed.trajectory never collide. The chunks are
merged at the end.

Output (<out> = ~/skoltech/ms_thesis/experiments/ibuprofen/mutation_RS by default):
    initial.traj                          the N generated structures
    connections                           ibuprofen bond list for GULP (1-based)
    <operator>/all_relaxed.trajectory     the mutated structures (same order as initial.traj)
    <operator>/POSCARS                    same, as POSCAR seeds
    chunks/chunk_XX/                      per-worker work dirs and log.txt

Run with ase_env (GULP paths from pyGulp configs/<EA_CONFIG>.yml, EA_CONFIG from .env):
    python scripts/experiments/ibuprofen_RS_mutation.py
    python scripts/experiments/ibuprofen_RS_mutation.py -n 300 --operators reaxff_small \
        --workers 16 --cpus 240-255 --min-dir <ibuprofen_top9_minima> --out <dir>
"""
import os

# one thread per worker: set before numpy is imported
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import multiprocessing as mp
import random
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
from ase.ga.data import PrepareDB
from ase.io import Trajectory, read
from pyxtal.symmetry import Group
from pyxtal.tolerance import Tol_matrix
from rdkit import Chem

from ea.core import mutation
from ea.structures.ase_creator import mol2ase2
from ea.structures.pyxtal_ea import compatible_groups, sym_molcrys_multi

warnings.filterwarnings('ignore')
MIN_DIR = Path.home() / 'skoltech/ms_thesis/experiments/ibuprofen/ibuprofen_top9_minima'
OPERATORS = {'reaxff': mutation.mutate_reaxff,
             'reaxff_small': mutation.mutate_reaxff_small,
             'reaxff_small2': mutation.mutate_reaxff_small2}


def parse_cpus(text):
    """'240-255' or '1,3,10-12' -> [240, ..., 255]"""
    cpus = []
    for part in text.split(','):
        a, _, b = part.partition('-')
        cpus += list(range(int(a), int(b or a) + 1))
    return cpus


def run_chunk(k, n, cpu, args, r_files, s_files, conn_path):
    """Generate n structures and mutate them in <out>/chunks/chunk_k."""
    wd0 = args.out / 'chunks' / f'chunk_{k:02d}'
    wd0.mkdir(parents=True, exist_ok=True)
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    log = open(wd0 / 'log.txt', 'w', buffering=1)
    sys.stdout = sys.stderr = log
    random.seed(args.seed * 1000 + k)
    np.random.seed(args.seed * 1000 + k)

    counts = [args.n_r, args.n_s]
    groups = compatible_groups(counts)
    tm = Tol_matrix(prototype='vdW', factor=args.vdw_factor)
    print(f'chunk {k}: {n} structures, cpu {cpu}, counts R/S {counts}, groups {groups}')
    structures = []
    for i in range(n):
        fr, fs = random.choice(r_files), random.choice(s_files)
        a = sym_molcrys_multi([mol2ase2(fr), mol2ase2(fs)], counts, groups=groups, tm=tm)
        a.info.update(mol_R=fr.stem, mol_S=fs.stem)
        structures.append(a)
        sg = a.info['spacegroup']
        print(f'{i + 1:3d}: SG {sg:3d} ({Group(sg).symbol:5s}) {fr.stem} + {fs.stem}')
    with Trajectory(wd0 / 'initial.traj', 'w') as t:
        for a in structures:
            t.write(a)

    for name in args.operators:
        wd = wd0 / name
        wd.mkdir(exist_ok=True)
        db = wd / 'theophilline.db'
        if db.exists():
            db.unlink()
        pdb = PrepareDB(str(db), simulation_cell=structures[0].copy(),
                        stoichiometry=structures[0].get_atomic_numbers())
        for a in structures:   # ids 2..n+1, as the operators expect
            pdb.add_unrelaxed_candidate(a.copy(), spacegroup=int(a.info['spacegroup']),
                                        mol_R=a.info['mol_R'], mol_S=a.info['mol_S'])
        print(f'\n===== {name} =====')
        op = OPERATORS[name](work_dir=str(wd), connection_dir=str(conn_path))
        op.mutate(n, poscar_name='POSCARS', keep_traj=True)
        print(f'{name}: {len(read(wd / op.traj_name, ":"))} structures')
    print('CHUNK DONE')


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--min-dir', type=Path, default=MIN_DIR, help='folder with mol_R/ and mol_S/')
    p.add_argument('-n', '--n-structures', type=int, default=30)
    p.add_argument('--n-r', type=int, default=2)
    p.add_argument('--n-s', type=int, default=2)
    p.add_argument('--vdw-factor', type=float, default=0.7)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--operators', nargs='+', default=list(OPERATORS), choices=list(OPERATORS))
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--cpus', type=parse_cpus, default=None,
                   help="CPUs to pin workers to, e.g. '240-255' (one per worker)")
    p.add_argument('--out', type=Path,
                   default=Path.home() / 'skoltech/ms_thesis/experiments/ibuprofen/mutation_RS')
    args = p.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.cpus is not None:
        assert len(args.cpus) >= args.workers, f'{args.workers} workers but only CPUs {args.cpus}'

    r_files = sorted((args.min_dir / 'mol_R').glob('min_*_R.mol'))
    s_files = sorted((args.min_dir / 'mol_S').glob('min_*_S.mol'))
    assert r_files and s_files, f'no MOL files in {args.min_dir}'

    # Bond list (same atom order and topology in all 18 files) -> GULP connections
    bonds = None
    for f in r_files + s_files:
        m = Chem.MolFromMolFile(str(f), removeHs=False)
        b = sorted((x.GetBeginAtomIdx() + 1, x.GetEndAtomIdx() + 1) for x in m.GetBonds())
        assert bonds is None or b == bonds, f'{f.name}: different bond list'
        bonds = b
    conn_path = args.out / 'connections'
    conn_path.write_text(''.join(f'connect {i:4d} {j:4d}\n' for i, j in bonds))

    sizes = [len(c) for c in np.array_split(np.arange(args.n_structures), args.workers)]
    print(f'{args.n_structures} structures in {args.workers} chunks {sizes}, CPUs {args.cpus}, '
          f'operators {args.operators}\nlogs: {args.out}/chunks/chunk_XX/log.txt', flush=True)
    ctx = mp.get_context('fork')
    procs = []
    for k, n in enumerate(sizes):
        cpu = args.cpus[k] if args.cpus is not None else None
        pr = ctx.Process(target=run_chunk, args=(k, n, cpu, args, r_files, s_files, conn_path))
        pr.start()
        procs.append(pr)
    for pr in procs:
        pr.join()
    failed = [k for k, pr in enumerate(procs) if pr.exitcode != 0]
    if failed:
        sys.exit(f'chunks {failed} failed, see their log.txt')

    # merge chunks in order
    chunk_dirs = [args.out / 'chunks' / f'chunk_{k:02d}' for k in range(len(sizes))]
    with Trajectory(args.out / 'initial.traj', 'w') as t:
        for c in chunk_dirs:
            for a in read(c / 'initial.traj', ':'):
                t.write(a)
    for name in args.operators:
        od = args.out / name
        od.mkdir(exist_ok=True)
        n_out = 0
        with Trajectory(od / 'all_relaxed.trajectory', 'w') as t, open(od / 'POSCARS', 'w') as pos:
            for c in chunk_dirs:
                for a in read(c / name / 'all_relaxed.trajectory', ':'):
                    t.write(a)
                    n_out += 1
                with open(c / name / 'POSCARS') as f:
                    shutil.copyfileobj(f, pos)
        print(f'{name}: {n_out} structures -> {od / "all_relaxed.trajectory"}')
    print(f'initial: {args.out / "initial.traj"}')


if __name__ == '__main__':
    main()
