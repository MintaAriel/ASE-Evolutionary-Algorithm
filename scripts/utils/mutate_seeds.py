from ea.core import mutation
import argparse

# work_dir = '/home/vito/uspex_matlab/theo_pyxtal/test_2'

def main():
    parser = argparse.ArgumentParser(
        description="mutation of seeds for molecular crystals"
    )

    parser.add_argument(
        "workdir",
        nargs="?",
        default=".",
        help="Working directory containing db with intial structures"
    )

    args = parser.parse_args()


    connection_dir = '/home/vito/PythonProjects/ASEProject/EA/data/theophylline/connections'

    transform = mutation.mutate_reaxff(work_dir=args.workdir,
                              connection_dir=connection_dir)

    transform.traj_name = 'prueba_2.traj'
    transform.mutate(20, poscar_name='POSCARS_2')


if __name__ == "__main__":
    main()
