from EA_SEARCH import Genetic_algorith
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run ASE calculations')
    parser.add_argument('--db', type=str, required=True,
                        help='Database file path')

    args = parser.parse_args()

    search = Genetic_algorith(args.db)
    search.start()





