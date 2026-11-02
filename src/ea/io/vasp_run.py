from ase.io import read, write
import subprocess
from ase.io import Trajectory
from pathlib import Path
import os
from ea.utils.config import load_config
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import shutil

cfg = load_config()
ms = cfg['vasp']
KPGEN = ms['kpeg']
VASP = ms['vasp6']
PHYSICAL_CORES = ms['cores']
NUMBERING = ms['cpu_numbering']
N = 2 * PHYSICAL_CORES  # total threads

def potcar_datasets(path):
    """Read complete PAW datasets in source order."""
    import re
    datasets = {}
    for block in re.findall(rb'.*?End of Dataset[^\n]*\n', Path(path).read_bytes(), flags=re.S):
        match = re.search(rb'TITEL\s*=\s*\S+\s+([A-Z][a-z]?)(?:_[^\s]+)?\s', block)
        if match is None:
            raise ValueError(f"Cannot identify POTCAR dataset in {path}")
        symbol = match.group(1).decode()
        if symbol in datasets:
            raise ValueError(f"Multiple potentials for {symbol} in {path}")
        datasets[symbol] = block
    if not datasets:
        raise ValueError(f"No complete POTCAR datasets in {path}")
    return datasets


def sort_poscar(atoms, order=None):
    """Group species in potential order or first-occurrence order."""
    symbols = list(dict.fromkeys(atoms.get_chemical_symbols()))
    order = symbols if order is None else list(order)
    missing = set(symbols) - set(order)
    if missing:
        raise ValueError(f"Missing potentials for elements: {sorted(missing)}")
    return atoms[sorted(range(len(atoms)), key=lambda i: order.index(atoms[i].symbol))]


def create_kpoints(cwd, args=None):
    if args is None:
        args = ["-g", "auto", "-d", "25"]
    elif isinstance(args, str):
        args = args.split()
    subprocess.run([KPGEN, *args], cwd=str(cwd), check=True)

def create_vasp_sp(input_dir, out_dir, traj, kpoints_args=None):
    files = os.listdir(input_dir)
    if 'INCAR' not in files:
        raise FileNotFoundError(f'No INCAR file to copy in {input_dir}')
    if 'POTCAR' not in files:
        raise FileNotFoundError(f'No POTCAR file to copy in {input_dir}')

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for i in range(len(traj)):
        calc_fold_path = Path(out_dir) / f'vasp_{i}'
        calc_fold_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(input_dir) / 'INCAR', calc_fold_path)
        datasets = potcar_datasets(Path(input_dir) / 'POTCAR')
        atoms = sort_poscar(traj[i], datasets)
        present = set(atoms.get_chemical_symbols())
        (calc_fold_path / 'POTCAR').write_bytes(
            b''.join(block for symbol, block in datasets.items() if symbol in present))
        write(calc_fold_path / 'POSCAR', atoms, format='vasp')
        create_kpoints(calc_fold_path, kpoints_args)

def cpu_affinity():
    if NUMBERING == 'adjacent':
        thread_pairs = [(2 * i, 2 * i + 1) for i in range(PHYSICAL_CORES)]
        #[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    elif NUMBERING == 'interleaved':
        thread_pairs = [(i, i + PHYSICAL_CORES) for i in range(PHYSICAL_CORES)]
        #[(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]

    return thread_pairs

def threads_to_use(threads, threads_per_job, mode):
    """
    threads = [low, high]  (inclusive range)
    mode 'a': allow SMT (use both threads per core)
    mode 'b': 1 thread per core
    """

    low, high = threads
    all_pairs = cpu_affinity()   # [(t0,t1), (t2,t3), ...]

    # -------------------------
    # Build filtered thread list
    # -------------------------
    if mode == 'a':
        # use BOTH SMT threads, but only those inside range
        flat = [
            t for pair in all_pairs
            for t in pair
            if low <= t <= high
        ]

    elif mode == 'b':
        # use only ONE thread per core (first element of pair)
        flat = [
            pair[0] for pair in all_pairs
            if low <= pair[0] <= high
        ]

    else:
        raise ValueError("mode must be 'a' or 'b'")

    # -------------------------
    # Chunk into jobs
    # -------------------------
    affinity = []
    for i in range(0, len(flat), threads_per_job):
        chunk = flat[i:i + threads_per_job]
        if len(chunk) == threads_per_job:
            affinity.append(chunk)

    return affinity


def run_vasp(job_id, thread_list, cwd, mode):
    """
    Run a single VASP job pinned to specific CPUs
    """

    thread_str = ",".join(map(str, thread_list))

    if mode == 'a':
        cmd = [VASP, "-a", "-c", thread_str, ]
    elif mode == 'b':
        cmd = [VASP, "-b", thread_str]
    else:
        raise ValueError("mode must be 'a' or 'b'")

    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True
    )

    if result.returncode:
        raise RuntimeError(f"VASP job {job_id} failed ({result.returncode}): {result.stderr}\n{result.stdout}")
    return job_id, result.stdout, result.stderr


def _run_vasp_leased(job_id, job_dir, mode, chunk_queue):
    """Lease a CPU chunk from the shared queue, run VASP, then return it."""
    cpu_chunk = chunk_queue.get()
    try:
        return run_vasp(job_id, cpu_chunk, job_dir, mode)
    finally:
        chunk_queue.put(cpu_chunk)


def run_vasp_folders(out_dir, traj, threads, threads_per_job, mode):

    affinity = threads_to_use(threads, threads_per_job, mode)

    manager = mp.Manager()
    chunk_queue = manager.Queue()
    for chunk in affinity:
        chunk_queue.put(chunk)

    jobs = []
    for i, structure in enumerate(traj):
        job_dir = Path(out_dir) / f"vasp_{i}"
        job_dir.mkdir(parents=True, exist_ok=True)
        jobs.append((i, job_dir))

    with ProcessPoolExecutor(max_workers=len(affinity)) as executor:

        futures = [
            executor.submit(_run_vasp_leased, job_id, job_dir, mode, chunk_queue)
            for job_id, job_dir in jobs
        ]

        for f in as_completed(futures):
            job_id, out, err = f.result()
            print(f"Job {job_id} finished")

def read_runs(traj, out_dir):
    atoms = []
    for i, _ in enumerate(traj):
        outcar_path = Path(out_dir) / f"vasp_{i}" / 'OUTCAR'
        atoms.append(read(outcar_path, format='vasp-out'))

    out_path = Path(out_dir) / 'results.traj'
    with Trajectory(out_path, mode='a') as traj_out:
        for a in atoms:
            traj_out.write(a)
    print(f'OUTCAR results saved to {out_path}')








