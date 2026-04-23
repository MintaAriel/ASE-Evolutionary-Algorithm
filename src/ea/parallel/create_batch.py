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
    coords, cells, types = build_batch_deepmd(batch_atoms_list, calculator.type_dict)
    E, F, V = calculator.dp.eval(coords, cells, types)[:3]

    energies = np.asarray(E).reshape(-1)
    forces_list, stress_voigt = [], []
    for k, at in enumerate(batch_atoms_list):
        forces_list.append(np.asarray(F[k]).reshape(-1, 3))
        virial = np.asarray(V[k]).reshape(3, 3)
        stress_3x3 = -virial / at.get_volume()
        stress_3x3 = 0.5 * (stress_3x3 + stress_3x3.T)
        stress_voigt.append(full_3x3_to_voigt_6_stress(stress_3x3))

    return energies, forces_list, stress_voigt


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




