from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.data import DataConnection
from ase.ga.offspring_creator import OperationSelector
from ase.ga.ofp_comparator import OFPComparator
from ase.ga.population import Population
from ase.ga.soft_mutation import SoftMutation
from ase.ga.standardmutations import StrainMutation
from ase.ga.utilities import CellBounds, closest_distances_generator
from ase.io import write
import os
from relax import Gulp_relaxation
from ase.ga.startgenerator import StartGenerator
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers



blocks=[('Mg', 4), ('Al', 8), ('O', 16)]
atom_num = [atomic_numbers[name] for (name, number) in blocks]
blmin = closest_distances_generator(atom_numbers=atom_num, ratio_of_covalent_radii=0.5)
cellbounds = CellBounds(
            bounds={
                'phi': [35, 145],
                'chi': [35, 145],
                'psi': [35, 145],
                'a': [3, 50],
                'b': [3, 50],
                'c': [3, 50],
            }
        )

splits = {(2,): 1, (1,): 0}

def trans_sym_generator(volume):
    sg = StartGenerator(
        slab=Atoms('', pbc=True),
        blocks=blocks,
        blmin= blmin,
        box_volume=volume,
        number_of_variable_cell_vectors=3,
        cellbounds=cellbounds,
        splits=splits,
    )
    a = sg.get_new_candidate()
    return a


class Genetic_algorith():
    def __init__(self, database ):
        self.db = database
        self.da = DataConnection(self.db)
        self.slab = self.da.get_slab()
        self.atoms_to_opt = self.da.get_atom_numbers_to_optimize()
        self.n_top = len(self.atoms_to_opt)
        self.blmin = closest_distances_generator(self.atoms_to_opt, 0.5)
        self.relax = Gulp_relaxation(self.db)

    def start(self):
        dir = os.path.dirname(self.db)
        # Use Oganov's fingerprint functions to decide whether two structures are identical or not
        comp = OFPComparator(
            n_top=self.n_top,
            dE=1.0,
            cos_dist_max=1e-3,
            rcut=10.0,
            binwidth=0.05,
            pbc=[True, True, True],
            sigma=0.05,
            nsigma=4,
            recalculate=False,
        )

        cellbounds = CellBounds(
            bounds={
                'phi': [20, 160],
                'chi': [20, 160],
                'psi': [20, 160],
                'a': [2, 60],
                'b': [2, 60],
                'c': [2, 60],
            }
        )

        # Define a pairing operator with 100% (0%) chance that the first
        # (second) parent will be randomly translated, and with each parent
        # contributing to at least 15% of the child's scaled coordinates
        pairing = CutAndSplicePairing(
            self.slab,
            self.n_top,
            self.blmin,
            p1=1.0,
            p2=0.0,
            minfrac=0.15,
            number_of_variable_cell_vectors=3,
            cellbounds=cellbounds,
            use_tags=False,
        )

        # Define a strain mutation with a typical standard deviation of 0.7
        # for the strain matrix elements (drawn from a normal distribution)
        strainmut = StrainMutation(
            self.blmin,
            stddev=0.7,
            cellbounds=cellbounds,
            number_of_variable_cell_vectors=3,
            use_tags=False,
        )

        # Define a soft mutation; we need to provide a dictionary with
        # (typically rather short) minimal interatomic distances which
        # is used to determine when to stop displacing the atoms along
        # the chosen mode. The minimal and maximal single-atom displacement
        # distances (in Angstrom) for a valid mutation are provided via
        # the 'bounds' keyword argument.
        blmin_soft = closest_distances_generator(self.atoms_to_opt, 0.1)
        softmut = SoftMutation(blmin_soft, bounds=[2.0, 5.0], used_modes_file=dir+'/used_modes.json', use_tags=False)
        # By default, the operator will update a "used_modes.json" file
        # after every mutation, listing which modes have been used so far
        # for each structure in the database. The mode indices start at 3
        # as the three lowest frequency modes are translational modes.

        # Set up the relative probabilities for the different operators
        operators = OperationSelector([3, 2, 1], [pairing, softmut, strainmut])

        #NOOLVIDES DE REGRESAR A 10 LA POBLACION
        max_generations = 30
        population_size = 20
        operator_percent = 0.8

        # Relax the initial candidates
        self.relax.relax_generation()

        population = Population(
            data_connection=self.da,
            population_size=population_size*operator_percent,
            comparator=comp,
            logfile=dir + f'/log.txt',
            use_extinct=True,
        )
        best_energies = []
        gen = 0
        all_max_indices = np.array([])


        while len(all_max_indices) <  max_generations and gen < 60:

            print(f'\n ====GENERATION {gen}====')
            energies = [x.info['key_value_pairs']['raw_score'] for x in population.pop]
            best_energies.append(max(energies))
            max_value = np.max(np.array(best_energies))
            all_max_indices = np.where( np.array(best_energies) == max_value)[0]


            print('Generations since the last minimum',len(all_max_indices))

            id = [x.info['confid'] for x in population.pop]

            # Calculate mean and standard deviation
            mu = np.mean(energies)
            sigma = np.std(energies)

            # Generate pseudo-energies for symmetric structures
            n_samples_sym = round(population_size * (1 - operator_percent))
            new_energies = np.random.normal(mu, sigma, n_samples_sym)

            print(energies)
            print('/n', id)

            #take best 20% volume of the population
            n_adapt = int(np.ceil(0.2 * len(population.pop)))
            v_new = np.mean([a.get_volume() for a in population.pop[:n_adapt]])
            print(v_new)
            #Create new random pseudo-symmetric structures
            fracRand = []

            for i in range(n_samples_sym):
                sym_struc = trans_sym_generator(v_new)
                sym_struc.info['key_value_pairs'] = {}
                sym_struc.info['data'] = {}
                sym_struc.info['key_value_pairs']['raw_score'] = new_energies[i]
                sym_struc.info['key_value_pairs']['generation'] = gen
                sym_struc.info['n_paired'] = 0
                sym_struc.info['looks_like'] = 0
                fracRand.append(sym_struc)

            self.da.add_more_relaxed_steps(fracRand)

            #modify the population to add the generated symmetric structures
            population.pop = population.pop + fracRand

            for step in range(population_size):
                print(f'Now starting configuration number {step+1}')

                # Create a new candidate
                a3 = None
                while a3 is None:
                    a1, a2 = population.get_two_candidates()
                    print(f"Using parents {a1.info['confid']} and {a2.info['confid']} ")

                    try:
                        a3, desc = operators.get_new_individual([a1, a2])
                        print(desc)
                    except Exception as error:
                        print('Error when creating offspring', error)

                a3.info['key_value_pairs']['generation'] = gen

                # Save the unrelaxed candidate
                self.da.add_unrelaxed_candidate(a3, description=desc)


                # Relax the new candidate and save it
                a3 = self.relax.use_gulp(a3)[0]
                print('The offspring has born')
                # If the relaxation has changed the cell parameters
                # beyond the bounds we disregard it in the population
                cell = a3.get_cell()
                if not cellbounds.is_within_bounds(cell):
                    self.da.kill_candidate(a3.info['confid'])
                    print('The offspring died')

            #We remove the pseudo-symmetric structures from the population
            population.pop  = population.pop[:int(population_size*operator_percent)]
            # Update the population
            population.update()
            current_pop = population.get_current_population()
            strainmut.update_scaling_volume(current_pop, w_adapt=0.5, n_adapt=4)
            pairing.update_scaling_volume(current_pop, w_adapt=0.5, n_adapt=4)
            traj_filename = os.path.join(dir, f'current_population{gen}.traj')
            write(traj_filename, current_pop)
            gen += 1

        print(f'\n ==================')
        print('\nTHE SEARCH IS DONE!')
