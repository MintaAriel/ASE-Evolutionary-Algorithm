"""Check ibuprofen R/S trajectories: per-molecule R/S label of the stereocentre, largest bond-length
deviation from the template (molecule intact?), minimum intermolecular contact, volume, space group.

Molecules = consecutive blocks of atoms of the template size (pyxtal and the mutation operators keep
this order). Each block is unwrapped along its bonds with the minimum-image convention (the operators
wrap atoms into the cell). The stereocentre (default atom 11, 1-based, as in the MOL files) is labelled
from the sign of the volume spanned by its substituents ranked by a first-sphere CIP rule
(COOH > aryl > CH3 > H), calibrated on the template, which must be an R molecule. No RDKit needed.

    python scripts/experiments/check_ibuprofen_RS.py --template mol_R/min_00_R.mol traj1 [traj2 ...]
"""
import argparse
from collections import deque
from pathlib import Path

import numpy as np
import spglib
from ase.geometry import find_mic
from ase.io import read

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument('trajs', nargs='+', type=Path)
p.add_argument('--template', type=Path, required=True, help='(R)-ibuprofen V2000 MOL file')
p.add_argument('--center', type=int, default=11, help='stereocentre atom (1-based)')
args = p.parse_args()


def read_mol(path):
    lines = Path(path).read_text().splitlines()
    c = next(i for i, l in enumerate(lines) if 'V2000' in l)
    na, nb = int(lines[c][0:3]), int(lines[c][3:6])
    pos = np.array([[float(l[0:10]), float(l[10:20]), float(l[20:30])] for l in lines[c + 1:c + 1 + na]])
    sym = [l[31:34].strip() for l in lines[c + 1:c + 1 + na]]
    bonds = np.array([(int(l[0:3]) - 1, int(l[3:6]) - 1) for l in lines[c + 1 + na:c + 1 + na + nb]])
    return pos, sym, bonds


ref, sym, bonds = read_mol(args.template)
nat = len(ref)
Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
nbrs = [[] for _ in range(nat)]
for i, j in bonds:
    nbrs[i].append(j)
    nbrs[j].append(i)
ref_len = np.linalg.norm(ref[bonds[:, 0]] - ref[bonds[:, 1]], axis=1)

center = args.center - 1
subs = nbrs[center]
assert len(subs) == 4, f'atom {args.center} has {len(subs)} neighbours'
key = lambda a: (Z[sym[a]], sorted((Z[sym[b]] for b in nbrs[a] if b != center), reverse=True))
ranked = sorted(subs, key=key, reverse=True)          # highest priority first
assert len({str(key(a)) for a in subs}) == 4, 'substituents not distinguishable in the first sphere'


def handedness(p):
    a, b, c, _ = (p[i] - p[center] for i in ranked)
    return np.sign(np.dot(a, np.cross(b, c)))


R_SIGN = handedness(ref)                              # template is R


def unwrap(pos, cell):
    out = pos.copy(); seen = {0}; q = deque([0])
    while q:
        i = q.popleft()
        for j in nbrs[i]:
            if j not in seen:
                d, _ = find_mic(pos[j] - pos[i], cell)
                out[j] = out[i] + d; seen.add(j); q.append(j)
    return out


for f in args.trajs:
    traj = read(f, ':')
    print(f'\n### {f}  ({len(traj)} structures)')
    print(f'{"#":>3} {"atoms":>5} {"SG":>6} {"V(A3)":>8} {"chir":>5} {"maxBondDev":>10} {"minContact":>10}')
    for k, a in enumerate(traj):
        nmol = len(a) // nat
        labs, dev = '', 0
        for m in range(nmol):
            pm = unwrap(a.positions[m * nat:(m + 1) * nat], a.cell)
            labs += 'R' if handedness(pm) == R_SIGN else 'S'
            ln = np.linalg.norm(pm[bonds[:, 0]] - pm[bonds[:, 1]], axis=1)
            dev = max(dev, np.abs(ln - ref_len).max())
        D = a.get_all_distances(mic=True)
        mol_id = np.repeat(np.arange(nmol), nat)
        cmin = D[mol_id[:, None] != mol_id[None]].min()
        sg = spglib.get_spacegroup((a.cell, a.get_scaled_positions(), a.numbers), symprec=0.01)
        print(f'{k:3d} {len(a):5d} {sg.split()[-1]:>6} {a.get_volume():8.1f} {labs:>5} {dev:10.3f} {cmin:10.2f}')
