#!/usr/bin/env python3
import os, shutil, argparse, subprocess, sys

from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from create_seeds_poscar import pyxtal_to_poscar

JOB_SH_CONTENT = """#!/bin/bash
export TMPDIR=/home/brian/tmp
mkdir -p $TMPDIR
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

/home/brian/PycharmProjects/ASEProject/uspex_python/uspex  >> log
"""

def copy_item(src: Path, dst: Path):
    """Copy file or directory if it exists."""
    if not src.exists():
        return
    if src.is_dir():
        # merge copy (Python 3.8+)
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

def make_job_sh(dst_dir: Path, overwrite: bool = True):
    job = dst_dir / "job.sh"
    if overwrite or not job.exists():
        job.write_text(JOB_SH_CONTENT, encoding="utf-8")
        os.chmod(job, 0o755)

def create_runs(template_dir: Path, runs_dir: Path, count: int):
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Accept both capitalizations for the USPEX input
    candidates_input = [template_dir / "INPUT.txt", template_dir / "Input.text"]
    input_src = next((p for p in candidates_input if p.exists()), None)
    if input_src is None:
        print("[WARN] Neither INPUT.txt nor Input.text found in template.", file=sys.stderr)

    for i in range(1, count + 1):
        rd = runs_dir / f"run_{i:03d}"
        rd.mkdir(parents=True, exist_ok=True)

        # Copy common USPEX artifacts if present
        copy_item(template_dir / "Seeds", rd / "Seeds")
        initial_pop = pyxtal_to_poscar(blocks = [('Mg', 4), ('Al',8),('O', 16)],
                                       amount=40,
                                       directory=str(rd/"Seeds"))
        initial_pop.create_poscars()
        copy_item(template_dir / "Specific",  rd / "Specific")
        copy_item(template_dir / "Submission", rd / "Submission")
        if input_src is not None:
            copy_item(input_src, rd / input_src.name)

        # job.sh (as given)
        make_job_sh(rd, overwrite=True)

    print(f"[OK] Created {count} run folders under: {runs_dir}")

def write_launch_all(runs_dir: Path, default_max_parallel: int = 50):
    """Create a small xargs-based launcher with adjustable concurrency."""
    sh = runs_dir.parent / "launch_all.sh"
    sh_content = f"""#!/usr/bin/env bash
set -euo pipefail
MAX_PARALLEL="${{1:-{default_max_parallel}}}"

# Run each job in its own directory with up to $MAX_PARALLEL in parallel.
# Adjust MAX_PARALLEL to fit your CPU: concurrency × threads_per_job ≤ total_cores.
find "{runs_dir}" -maxdepth 1 -type d -name 'run_*' | sort | \\
  xargs -n1 -P "$MAX_PARALLEL" -I{{}} bash -lc 'cd "{{}}" && ./job.sh'
"""
    sh.write_text(sh_content, encoding="utf-8")
    os.chmod(sh, 0o755)
    print(f"[OK] Wrote launcher: {sh}  (usage: ./launch_all.sh 50)")

def launch_now(runs_dir: Path, max_parallel: int):
    """Pure Python launcher with bounded concurrency."""
    # Gather run dirs
    run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("run_")])
    if not run_dirs:
        print("[ERR] No run_* folders found. Create them first.", file=sys.stderr)
        sys.exit(1)

    procs = []
    idx = 0

    def spawn(p: Path):
        # Run ./job.sh in the run directory
        return subprocess.Popen(
            ["bash", "-lc", "./job.sh"],
            cwd=p,
            stdout=open(p / "launcher.out", "ab"),
            stderr=subprocess.STDOUT,
        )

    # Start initial batch
    while idx < len(run_dirs) and len(procs) < max_parallel:
        procs.append((run_dirs[idx], spawn(run_dirs[idx])))
        idx += 1

    # Refill as they finish
    import time
    while procs:
        new_procs = []
        for rd, pr in procs:
            ret = pr.poll()
            if ret is None:
                new_procs.append((rd, pr))
            else:
                if ret != 0:
                    print(f"[WARN] {rd.name} exited with code {ret} (see {rd/'launcher.out'})")
                if idx < len(run_dirs):
                    new_procs.append((run_dirs[idx], spawn(run_dirs[idx])))
                    idx += 1
        procs = new_procs
        time.sleep(1)

    print("[OK] All jobs finished (launcher).")

def main():
    ap = argparse.ArgumentParser(description="Prepare and (optionally) launch many USPEX runs.")
    ap.add_argument("--template", default=".", help="Template directory (has Seeds, INPUT.txt, job.sh)")
    ap.add_argument("--runs-dir", default="runs", help="Where to put run_### folders")
    ap.add_argument("--count", type=int, default=100, help="How many runs to create")
    ap.add_argument("--make-only", action="store_true", help="Only create folders; do not launch")
    ap.add_argument("--launch", action="store_true", help="Launch jobs after creating folders")
    ap.add_argument("--max-parallel", type=int, default=50, help="Max concurrent jobs if launching")
    args = ap.parse_args()

    template_dir = Path(args.template).resolve()
    runs_dir = Path(args.runs_dir).resolve()

    create_runs(template_dir, runs_dir, args.count)
    write_launch_all(runs_dir, default_max_parallel=args.max_parallel)

    if args.launch and not args.make_only:
        print(f"[RUN] Launching with max_parallel={args.max_parallel}")
        launch_now(runs_dir, args.max_parallel)

if __name__ == "__main__":
    main()
