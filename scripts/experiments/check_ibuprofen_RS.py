"""Check ibuprofen R/S trajectories: per-molecule CIP label of the stereocentre, largest bond-length
deviation from the template (molecule intact?), minimum intermolecular contact, volume, space group.

Molecules = consecutive blocks of 33 atoms (pyxtal and the mutation operators keep this
order). Each block is unwrapped along its bonds with the minimum-image convention (the
operators wrap atoms into the cell); the CIP label then comes from RDKit on those 3D
coordinates, which does not depend on the conformer.

    python scripts/experiments/check_ibuprofen_RS.py ~/skoltech/ms_thesis/experiments/ibuprofen/mutation_RS/initial.traj ...
"""
import sys
from collections import deque
from pathlib import Path

import numpy as np
import spglib
from ase.geometry import find_mic
from ase.io import read
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

TEMPLATE = Path.home() / 'skoltech/ms_thesis/experiments/ibuprofen/ibuprofen_top9_minima/mol_R/min_00_R.mol'
tmol = Chem.MolFromMolFile(str(TEMPLATE), removeHs=False)
ref = tmol.GetConformer().GetPositions()
nat = len(ref)
nbrs = [[n.GetIdx() for n in a.GetNeighbors()] for a in tmol.GetAtoms()]
bonds = np.array([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in tmol.GetBonds()])
ref_len = np.linalg.norm(ref[bonds[:, 0]] - ref[bonds[:, 1]], axis=1)


def unwrap(pos, cell):
    out = pos.copy(); seen = {0}; q = deque([0])
    while q:
        i = q.popleft()
        for j in nbrs[i]:
            if j not in seen:
                d, _ = find_mic(pos[j] - pos[i], cell)
                out[j] = out[i] + d; seen.add(j); q.append(j)
    return out


def cip(p):
    m = Chem.Mol(tmol)
    conf = m.GetConformer()
    for i, xyz in enumerate(p):
        conf.SetAtomPosition(i, xyz.tolist())
    Chem.AssignStereochemistryFrom3D(m)
    rdCIPLabeler.AssignCIPLabels(m)
    return ''.join(a.GetProp('_CIPCode') for a in m.GetAtoms() if a.HasProp('_CIPCode')) or '?'


for f in sys.argv[1:]:
    traj = read(f, ':')
    print(f'\n### {f}  ({len(traj)} structures)')
    print(f'{"#":>3} {"atoms":>5} {"SG":>4} {"V(A3)":>8} {"chir":>5} {"maxBondDev":>10} {"minContact":>10}')
    for k, a in enumerate(traj):
        nmol = len(a) // nat
        labs, rms, mols = '', 0, []
        for m in range(nmol):
            p = unwrap(a.positions[m * nat:(m + 1) * nat], a.cell)
            mols.append(p)
            labs += cip(p)
            ln = np.linalg.norm(p[bonds[:, 0]] - p[bonds[:, 1]], axis=1)
            rms = max(rms, np.abs(ln - ref_len).max())
        D = a.get_all_distances(mic=True)
        mol_id = np.repeat(np.arange(nmol), nat)
        cmin = D[mol_id[:, None] != mol_id[None]].min()
        sg = spglib.get_spacegroup((a.cell, a.get_scaled_positions(), a.numbers), symprec=0.01)
        print(f'{k:3d} {len(a):5d} {sg.split()[-1]:>6} {a.get_volume():8.1f} {labs:>5} {rms:10.3f} {cmin:10.2f}')
