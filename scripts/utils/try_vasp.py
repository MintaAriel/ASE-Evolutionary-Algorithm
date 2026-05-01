import argparse

from ase.io import read

from ea.io.vasp_run import create_vasp_sp, run_vasp_folders, read_runs


def parse_threads(s):
    parts = s.replace(',', ' ').split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "threads must be two ints, e.g. '0,3' or '0 3'"
        )
    return [int(parts[0]), int(parts[1])]


def main():
    parser = argparse.ArgumentParser(
        description="Run parallel VASP single-points over an ASE trajectory."
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Output directory (must contain INCAR and POTCAR; per-structure "
             "subfolders vasp_0..vasp_N will be created here).",
    )
    parser.add_argument(
        "--traj-dir", required=True,
        help="Path to the ASE trajectory file to read.",
    )
    parser.add_argument(
        "--threads", required=True, type=parse_threads,
        help="Inclusive logical-CPU range, e.g. '0,3' or '0 3'.",
    )
    parser.add_argument(
        "--threads-per-job", required=True, type=int,
        help="Number of MPI ranks per VASP job.",
    )
    parser.add_argument(
        "--mode", required=True, choices=("a", "b"),
        help="'a' = use both SMT threads of each core; "
             "'b' = one rank per physical core (no SMT sharing).",
    )
    parser.add_argument(
        "-n", "--num", type=int, default=None,
        help="Take only the first N frames of the trajectory "
             "(default: all frames).",
    )
    args = parser.parse_args()

    traj = read(args.traj_dir, index=":")
    if args.num is not None:
        if args.num < 1:
            parser.error("--num must be >= 1")
        traj = traj[:args.num]

    create_vasp_sp(args.out_dir, args.out_dir, traj)
    run_vasp_folders(
        args.out_dir, traj, args.threads, args.threads_per_job, args.mode,
    )
    read_runs(traj, args.out_dir)


if __name__ == "__main__":
    main()
