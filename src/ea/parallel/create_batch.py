import numpy as np



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

