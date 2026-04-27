import atexit
import concurrent.futures
import multiprocessing as mp

import numpy as np
from ase.stress import full_3x3_to_voigt_6_stress


def build_batch_deepmd(crystals, type_dict):
    '''
    This batch is only created if the crystals have the same amount of atoms in the same
    order (type)
    :param crystals: Ase atoms list
    :param type_dict: DP calc.type_dict
    :return:
    '''
    coords = []
    cells = []
    atypes = []

    for c in crystals:
        coords.append(c.get_positions().reshape(-1))
        cells.append(c.get_cell().reshape(-1))


    coords = np.stack(coords)   # (B, N*3)
    cells  = np.stack(cells)    # (B, 9)
    atypes = np.array(atypes)   # ( N,)

    # atom types ONLY from first structure
    symbols = crystals[0].get_chemical_symbols()
    atypes = np.array([type_dict[s] for s in symbols])  # (N,)

    return coords, cells, atypes


def batch_calculator_deepmd(batch_atoms_list, calculator):
    """Batched evaluator for a deepmd `DP` calculator.

    Returns (energies, forces_list, stress_voigt_list) in the shape expected
    by ParallelFIRE / ParallelLBFGS:
      - energies:          ndarray shape (B,)
      - forces_list:       list of B arrays, each (N_i, 3)
      - stress_voigt_list: list of B arrays, each (6,) in ASE convention

    DP's virial v satisfies stress = -v / volume (ASE sign convention).
    """
    import time

    # ---- diagnostic: structure layout ----
    atom_counts = [len(a) for a in batch_atoms_list]
    symbol_sigs = [tuple(a.get_chemical_symbols()) for a in batch_atoms_list]
    unique_counts = sorted(set(atom_counts))
    unique_sigs = len(set(symbol_sigs))
    print(f"[batch_calc] B={len(batch_atoms_list)}  unique_atom_counts={unique_counts}  "
          f"unique_symbol_layouts={unique_sigs}")
    if len(unique_counts) > 1 or unique_sigs > 1:
        print("[batch_calc] WARNING: heterogeneous batch — build_batch_deepmd assumes "
              "all structures share crystals[0]'s symbols/length. eval() may error or "
              "silently fall back to per-structure evaluation.")

    coords, cells, types = build_batch_deepmd(batch_atoms_list, calculator.type_dict)

    abs_obj = calculator.dp.deep_eval.auto_batch_size
    print(f"[batch_calc] BEFORE eval  current={abs_obj.current_batch_size}  "
          f"max_working={abs_obj.maximum_working_batch_size}  "
          f"min_not_working={abs_obj.minimal_not_working_batch_size}")

    # abs_obj.current_batch_size = 128
    print(f"[batch_calc] forced current_batch_size=128")

    t0 = time.perf_counter()
    E, F, V = calculator.dp.eval(coords, cells, types)[:3]
    dt = time.perf_counter() - t0

    print(f"[batch_calc] AFTER eval   current={abs_obj.current_batch_size}  "
          f"max_working={abs_obj.maximum_working_batch_size}  "
          f"min_not_working={abs_obj.minimal_not_working_batch_size}")
    print(f"[batch_calc] eval() wall time: {dt:.3f}s  ({dt/len(batch_atoms_list)*1000:.1f} ms/struct)")

    energies = np.asarray(E).reshape(-1)
    forces_list, stress_voigt = [], []
    for k, at in enumerate(batch_atoms_list):
        forces_list.append(np.asarray(F[k]).reshape(-1, 3))
        virial = np.asarray(V[k]).reshape(3, 3)
        stress_3x3 = -virial / at.get_volume()
        stress_3x3 = 0.5 * (stress_3x3 + stress_3x3.T)
        stress_voigt.append(full_3x3_to_voigt_6_stress(stress_3x3))

    return energies, forces_list, stress_voigt


# Persistent multiprocessing pool for DeepMD parallel evaluation.
# Each worker process owns its own DP instance and CUDA context, giving the
# same parallelism profile as N separate Python processes. Pool is built lazily
# on first use, keyed on (n_workers, model_path, device); rebuilt if the key
# changes; torn down at interpreter exit.
_dp_pool = None
_dp_pool_key = None


def _dp_worker_init(model_path: str, device: str) -> None:
    """Run once per worker process at startup. Loads one DP into the worker."""
    global _worker_dp, _worker_type_dict
    from deepmd.calculator import DP
    _worker_dp = DP(model=model_path, device=device)
    # Pin auto_batch_size to 1 — each task is a single frame, no need to
    # search for a larger batch (we never call eval with multiple frames).
    _worker_dp.dp.deep_eval.auto_batch_size.current_batch_size = 1
    _worker_type_dict = _worker_dp.type_dict


def _dp_eval_chunk(chunk):
    """Evaluate a list of (orig_idx, atoms) tuples inside a worker process.

    Returns a list of (orig_idx, energy, forces, stress_voigt) tuples so the
    caller can scatter back into the original batch order.
    """
    global _worker_dp, _worker_type_dict
    out = []
    for idx, atoms in chunk:
        coords = atoms.get_positions().reshape(1, -1)
        cells = atoms.get_cell().reshape(1, -1)
        types = np.array([_worker_type_dict[s] for s in atoms.get_chemical_symbols()])
        E, F, V = _worker_dp.dp.eval(coords, cells, types)[:3]
        energy = float(np.asarray(E).reshape(-1)[0])
        forces = np.asarray(F[0]).reshape(-1, 3)
        virial = np.asarray(V[0]).reshape(3, 3)
        stress_3x3 = -virial / atoms.get_volume()
        stress_3x3 = 0.5 * (stress_3x3 + stress_3x3.T)
        out.append((idx, energy, forces, full_3x3_to_voigt_6_stress(stress_3x3)))
    return out


def _shutdown_dp_pool() -> None:
    global _dp_pool, _dp_pool_key
    if _dp_pool is not None:
        try:
            _dp_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        _dp_pool = None
        _dp_pool_key = None


atexit.register(_shutdown_dp_pool)


def _ensure_dp_pool(n_workers: int, model_path: str, device: str):
    global _dp_pool, _dp_pool_key
    key = (n_workers, str(model_path), device)
    if _dp_pool is None or _dp_pool_key != key:
        _shutdown_dp_pool()
        ctx = mp.get_context('spawn')  # spawn required: fork breaks CUDA contexts
        _dp_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_dp_worker_init,
            initargs=(str(model_path), device),
        )
        _dp_pool_key = key
    return _dp_pool


def batch_calculator_deepmd_mp(batch_atoms_list, calculator, n_workers=4):
    """Multiprocessing per-structure evaluator for DeepMD `DP`.

    Splits the batch into n_workers chunks and dispatches each to a persistent
    worker process. Each worker owns its own DP instance and CUDA context, so
    `dp.eval` calls run with the same isolation as N separate Python processes
    — sidestepping the GIL and single-context cuBLAS/allocator serialization
    that defeated the threaded version.

    The pool is built on first call and reused thereafter, so the (large)
    spawn + DP-load cost is paid once per Python session.

    Results are returned in the original batch order.
    """
    import time

    B = len(batch_atoms_list)
    if B == 0:
        return np.empty(0), [], []

    # Pull model path + device from the provided calculator.
    try:
        model_path = calculator.dp.deep_eval.model_path
    except AttributeError:
        model_path = calculator.parameters.get('model')
    if model_path is None:
        raise RuntimeError(
            "Could not extract model path from calculator; expected "
            "calculator.dp.deep_eval.model_path or calculator.parameters['model']."
        )
    try:
        device_type = next(calculator.dp.deep_eval.model.parameters()).device.type
        device = 'gpu' if device_type == 'cuda' else 'cpu'
    except Exception:
        device = 'gpu'

    n_workers = max(1, min(n_workers, B))

    pool_was_cold = _dp_pool is None or _dp_pool_key != (n_workers, str(model_path), device)
    t_pool = time.perf_counter()
    pool = _ensure_dp_pool(n_workers, model_path, device)
    if pool_was_cold:
        print(f"[batch_calc_mp] spawning {n_workers} worker processes on {device} "
              f"(model load happens lazily on first task)...")

    chunks = [list(idxs) for idxs in np.array_split(np.arange(B), n_workers)]
    chunks = [c for c in chunks if len(c) > 0]
    payloads = [[(int(i), batch_atoms_list[int(i)]) for i in c] for c in chunks]

    energies = np.empty(B)
    forces_list = [None] * B
    stress_voigt = [None] * B

    t0 = time.perf_counter()
    futures = [pool.submit(_dp_eval_chunk, p) for p in payloads]
    for fut in concurrent.futures.as_completed(futures):
        for idx, energy, forces, stress in fut.result():
            energies[idx] = energy
            forces_list[idx] = forces
            stress_voigt[idx] = stress
    dt = time.perf_counter() - t0
    print(f"[batch_calc_mp] B={B} workers={len(chunks)}  "
          f"wall={dt:.3f}s  ({dt / B * 1000:.1f} ms/struct)"
          + (f"  (cold-start incl. worker init: {time.perf_counter() - t_pool:.2f}s)"
             if pool_was_cold else ""))

    return energies, forces_list, stress_voigt


# Backward-compatible alias — the old threaded function name still works,
# but now dispatches to the multiprocessing implementation.
batch_calculator_deepmd_threaded = batch_calculator_deepmd_mp


def batch_calculator_uma(batch_atoms_list, calculator):
    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.datasets import data_list_collater

    number_atoms = [len(a) for a in batch_atoms_list]

    data_list = [
        AtomicData.from_ase(
            atoms,
            task_name="omat",
            r_edges=6.0,
            max_neigh=50,
            radius=6.0,
        )
        for atoms in batch_atoms_list
    ]

    batch = data_list_collater(data_list, otf_graph=True)

    pred = calculator.predictor.predict(batch)

    energies = pred["energy"].detach().cpu().numpy()
    forces = pred["forces"].detach().cpu().numpy()
    stress = pred['stress'].detach().cpu().numpy()
    stress_voigt = []

    for i_stress in stress:
        stress_3x3 =  i_stress.reshape(3, 3)
        stress_voigt.append(full_3x3_to_voigt_6_stress(stress_3x3))


    split_indices = np.cumsum(number_atoms)[:-1]
    forces_list = np.split(forces, split_indices)


    return energies, forces_list, stress_voigt




