from ase.io import read, Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from matplotlib.figure import Figure
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import copy
from queue import Queue
import json
import os
import shutil
import numpy as np


def relax_traj(relax, traj_path, out_path, index=':'):
    """Relax every structure in traj_path with GULP and write the optimized ones to out_path.

    relax is a pygulp relaxation object exposing use_gulp(atoms), e.g. Gulp_relaxation_noadd.
    Returns a list of (structure index, energy in eV) for the successful relaxations.
    """
    traj_path = Path(traj_path)
    structures = read(traj_path, index=index)
    energies = []

    with Trajectory(out_path, 'w') as out_traj:
        for i, atoms in enumerate(structures):
            print(f'[{i + 1}/{len(structures)}] relaxing {traj_path.name}')
            new_crys = relax.use_gulp(atoms)

            # use_gulp returns the input structure when GULP fails, so check the results
            results = new_crys.calc.results if new_crys.calc is not None else {}
            if 'energy' not in results:
                print(f'  structure {i} failed, skipping')
                continue

            energy = results['energy']
            n_molecules = len(np.unique(new_crys.get_tags()))
            new_crys.info['source_index'] = i
            new_crys.info['energy_per_molecule'] = energy / n_molecules

            # Store the energy without keeping the GULP calculator attached
            new_crys.calc = SinglePointCalculator(new_crys, energy=energy)
            out_traj.write(new_crys)
            energies.append((i, energy))
            print(f'  E = {energy:.4f} eV')

    return energies


def relax_trajs_parallel(relax, traj_paths, out_paths, n_jobs=1,
                         cpu_pins=None, index=':', collect_only=False):
    """Relax trajectories with one shared pool of single-threaded GULP jobs.

    ``cpu_pins`` optionally assigns one Linux CPU ID to each worker. Every
    frame has its own calculation directory beside its output trajectory.
    Results are collected as jobs finish and written in source order. Returns
    one list of (source index, energy in eV) pairs per input trajectory.
    A sibling ``*_status.json`` records unsuccessful and unconverged frames.
    Like ``relax_traj``, finite-energy results are kept even if unconverged;
    ``gulp_converged`` metadata identifies them. With ``collect_only=True``,
    parse retained GULP outputs without running or deleting calculations.
    """
    traj_paths = [Path(p).resolve() for p in traj_paths]
    out_paths = [Path(p).resolve() for p in out_paths]
    if not traj_paths or len(traj_paths) != len(out_paths):
        raise ValueError('Provide matching, nonempty input and output paths')
    if len(set(out_paths)) != len(out_paths) or set(out_paths) & set(traj_paths):
        raise ValueError('Output paths must be distinct and differ from all inputs')
    if n_jobs < 1:
        raise ValueError('n_jobs must be positive')
    if cpu_pins is not None:
        cpu_pins = list(cpu_pins)
        if len(cpu_pins) != n_jobs or len(set(cpu_pins)) != n_jobs:
            raise ValueError('cpu_pins must contain one distinct CPU per worker')
        if not hasattr(os, 'sched_setaffinity'):
            raise ValueError('CPU pinning requires sched_setaffinity')
        unavailable = set(cpu_pins) - os.sched_getaffinity(0)
        if unavailable:
            raise ValueError(f'Unavailable CPUs: {sorted(unavailable)}')

    batches = []
    for path in traj_paths:
        batch = read(path, index=index)
        batches.append([batch] if hasattr(batch, 'get_positions') else batch)
    results = [[None] * len(batch) for batch in batches]
    failures = [[] for _ in batches]
    for path in out_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    # External GULP processes inherit the environment and their calling thread's
    # CPU affinity. Only the coordinator writes the combined trajectories.
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    slots = Queue()
    for worker in range(n_jobs):
        slots.put(cpu_pins[worker] if cpu_pins is not None else None)

    def task(batch_id, frame_id, atoms):
        pin = slots.get()
        previous_affinity = None
        try:
            if pin is not None:
                previous_affinity = os.sched_getaffinity(0)
                os.sched_setaffinity(0, {pin})
            path = out_paths[batch_id]
            frame_dir = path.parent / f'{path.stem}_work' / f'frame_{frame_id:04d}'
            calc_dir = frame_dir / 'CalcFold'
            calc_dir.mkdir(parents=True, exist_ok=True)
            # A repeated run must not read an old GULP result after a failure.
            if not collect_only:
                for name in ('ginput1.got', 'out1.cif'):
                    (calc_dir / name).unlink(missing_ok=True)
            worker_relax = copy(relax)
            worker_relax.path = frame_dir
            if getattr(relax, 'library', None) is not None:
                library = Path(relax.library)
                source = Path(relax.path) / 'CalcFold' / library
                if not source.is_file():
                    source = Path(relax.gulp_lib_dir) / library
                shutil.copy2(source, calc_dir / library.name)
            atoms = atoms.copy()
            atoms.calc = None
            if collect_only:
                got_path = calc_dir / 'ginput1.got'
                if not got_path.is_file() or 'Job Finished' not in got_path.read_text():
                    raise RuntimeError('No finished GULP output to collect')
                calc = worker_relax.number_of_optmi(
                    worker_relax.gulp_keywords, worker_relax.gulp_options,
                    atoms, worker_relax.library)
                calc.prefix = 'ginput1'
                calc.directory = str(calc_dir)
                calc.atoms = atoms.copy()
                calc.read_results()
                optimized = calc.atoms.copy()
                optimized.calc = calc
            else:
                optimized = worker_relax.use_gulp(atoms)
            calc = optimized.calc
            energy = calc.results.get('energy') if calc is not None else None
            if energy is None or not np.isfinite(energy):
                raise RuntimeError('GULP returned no finite energy; see CalcFold output')
            energy = float(energy)
            optimized.info['source_index'] = frame_id
            optimized.info['gulp_converged'] = bool(getattr(calc, 'optimized', True))
            optimized.info['energy_per_molecule'] = (
                energy / len(np.unique(optimized.get_tags())))
            optimized.calc = SinglePointCalculator(optimized, energy=energy)
            return optimized
        finally:
            if previous_affinity is not None:
                os.sched_setaffinity(0, previous_affinity)
            slots.put(pin)

    total = sum(map(len, batches))
    completed = 0
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        # Interleave inputs so both populations start using the shared pool.
        futures = {executor.submit(task, b, i, batch[i]): (b, i)
                   for i in range(max(map(len, batches), default=0))
                   for b, batch in enumerate(batches) if i < len(batch)}
        for future in as_completed(futures):
            b, i = futures[future]
            completed += 1
            try:
                results[b][i] = future.result()
                detail = f'E = {results[b][i].get_potential_energy():.6f} eV'
                if not results[b][i].info['gulp_converged']:
                    detail += ' (not converged)'
            except Exception as exc:
                detail = f'{type(exc).__name__}: {exc}'
                failures[b].append({'source_index': i, 'error': detail})
            print(f'[{completed}/{total}] {traj_paths[b].name} frame {i}: {detail}',
                  flush=True)

    all_energies = []
    for b, path in enumerate(out_paths):
        energies = []
        with Trajectory(path, 'w') as output:
            for i, atoms in enumerate(results[b]):
                if atoms is not None:
                    output.write(atoms)
                    energies.append((i, atoms.get_potential_energy()))
        status = {'input': str(traj_paths[b]), 'output': str(path),
                  'total': len(batches[b]), 'succeeded': len(energies),
                  'failed': len(failures[b]),
                  'unconverged': [i for i, atoms in enumerate(results[b])
                                  if atoms is not None and not atoms.info['gulp_converged']],
                  'failures': sorted(failures[b], key=lambda item: item['source_index'])}
        path.with_name(f'{path.stem}_status.json').write_text(
            json.dumps(status, indent=2) + '\n')
        all_energies.append(energies)
    return all_energies


def relax_traj_parallel(relax, traj_path, out_path, n_jobs=1,
                        cpu_pins=None, index=':', collect_only=False):
    """Single-trajectory wrapper for :func:`relax_trajs_parallel`."""
    return relax_trajs_parallel(relax, [traj_path], [out_path], n_jobs=n_jobs,
                               cpu_pins=cpu_pins, index=index,
                               collect_only=collect_only)[0]


def save_energy_distribution(energies, out_prefix, bins=20):
    """Save the energies to out_prefix.csv and a histogram to out_prefix.png."""
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(energies, dtype=float).reshape(-1, 2)

    np.savetxt(out_prefix.with_suffix('.csv'), data, delimiter=',',
               header='source_index,energy_eV', fmt=['%d', '%.6f'], comments='')

    # Figure without pyplot works headless and doesn't change the caller's backend
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    ax.hist(data[:, 1], bins=bins, edgecolor='black')
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Count')
    ax.set_title(f'{out_prefix.name} (n = {len(data)})')
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix('.png'), dpi=200)
