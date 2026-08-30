import os
import multiprocessing as mp

from dataclasses import dataclass
from pathlib import Path

from ase.ga.data import DataConnection
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE, LBFGS


TEST_DIR = '/home/sirena/Brian/THP_8mol/test2/theophilline.db'

MODEL = '/home/sirena/Brian/Github/ASE-Evolutionary-Algorithm/models/dpa3-pbed3-pytorch.pth'

OUT_DIR = Path(
    '/home/sirena/Brian/Github/ASE-Evolutionary-Algorithm/tests/gpu_multipro'
)


@dataclass
class RelaxConfig:

    model_key: str = "base_deepmd"

    fire_fmax: float = 0.10
    fire_steps: int = 500

    lbfgs_stages: tuple = (
        0.03,
        0.01,
        0.005,
        0.002,
        0.001
    )

    lbfgs_steps: int = 1200

    maxstep: float = 0.03

    threads: int = 2

    cores: list[int] | None = None


class DeepMDRelaxation:

    def __init__(self, config: RelaxConfig):

        self.cfg = config

    def run(self, atoms, out_dir):

        log_path = out_dir / "relax.log"
        traj_path = out_dir / "opt.traj"

        # FIRE

        fire = FIRE(
            FrechetCellFilter(atoms),
            logfile=str(log_path),
            trajectory=str(traj_path),
            maxstep=self.cfg.maxstep,
        )

        fire.run(
            fmax=self.cfg.fire_fmax,
            steps=self.cfg.fire_steps
        )

        # LBFGS

        lbfgs = LBFGS(
            FrechetCellFilter(atoms),
            logfile=str(log_path),
            trajectory=str(traj_path),
            append_trajectory=True,
            maxstep=self.cfg.maxstep,
            memory=40,
        )

        for fmax in self.cfg.lbfgs_stages:

            lbfgs.run(
                fmax=fmax,
                steps=self.cfg.lbfgs_steps
            )

        energy = atoms.get_potential_energy()

        return {
            "optimizer": "fire_then_staged_lbfgs",
            "nsteps": int(fire.nsteps + lbfgs.nsteps),
            "lbfgs_stages": list(self.cfg.lbfgs_stages),
            "energy": energy,
        }


def worker(gpu_id, task_queue, result_queue):


    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Import CUDA-dependent library AFTER setting CUDA_VISIBLE_DEVICES

    from deepmd.calculator import DP

    print(
        f"Worker PID={os.getpid()} "
        f"assigned to GPU {gpu_id}"
    )

    # --------------------------------------------------
    # Load DeepMD ONCE
    # --------------------------------------------------

    calc = DP(
        model=MODEL,
        device="gpu"
    )

    deep = DeepMDRelaxation(
        RelaxConfig()
    )

    da = DataConnection(TEST_DIR)

    # --------------------------------------------------
    # Process structures until queue is empty
    # --------------------------------------------------

    while True:

        structure_id = task_queue.get()

        if structure_id is None:
            break

        print(
            f"GPU {gpu_id} | "
            f"processing structure {structure_id}"
        )

        atom = da.get_atoms(structure_id)

        atom.calc = calc

        out_dir = (
            OUT_DIR /
            f"structure_{structure_id}"
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        result = deep.run(
            atom,
            out_dir=out_dir
        )

        result_queue.put(
            (
                structure_id,
                result
            )
        )

        print(
            f"GPU {gpu_id} | "
            f"finished structure {structure_id}"
        )


if __name__ == "__main__":

    mp.set_start_method("spawn")

    # GPUs we want to use

    gpu_ids = [0, 1, 2]

    # Structures 2 through 13

    structure_ids = range(2, 14)

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # --------------------------------------------------
    # Put all structures into the queue
    # --------------------------------------------------

    for structure_id in structure_ids:

        task_queue.put(structure_id)

    # One termination signal per worker

    for _ in gpu_ids:

        task_queue.put(None)

    # --------------------------------------------------
    # Start workers
    # --------------------------------------------------

    workers = []

    for gpu_id in gpu_ids:

        p = mp.Process(
            target=worker,
            args=(
                gpu_id,
                task_queue,
                result_queue
            )
        )

        p.start()

        workers.append(p)

    # --------------------------------------------------
    # Wait for workers
    # --------------------------------------------------

    for p in workers:

        p.join()

    # --------------------------------------------------
    # Collect results
    # --------------------------------------------------

    results = []

    while not result_queue.empty():

        structure_id, result = result_queue.get()

        results.append(
            (
                structure_id,
                result
            )
        )

    results.sort(
        key=lambda x: x[0]
    )

    print("\nResults:")

    for structure_id, result in results:

        print(
            f"Structure {structure_id}: "
            f"{result['energy']}"
        )


