"""Batched vibrational/ZPE calculations.

Mirrors ``ase.vibrations.Vibrations`` but performs the finite-difference
force evaluations for many displacements (and many structures) in a single
call to a user-supplied batch evaluator with the signature::

    batch_evaluator(atoms_list) -> (energies, forces_list, stress_voigt_list)

This is the same evaluator interface used by ``ParallelFIRE`` /
``ParallelLBFGS`` in this package, so the same wrappers around deepmd / UMA
calculators apply (see :func:`make_deepmd_evaluator`).
"""

import sys
from collections import defaultdict, namedtuple
from collections.abc import Callable
from typing import List, Optional, Sequence

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.vibrations.data import VibrationsData

from ea.parallel.create_batch import batch_calculator_deepmd


# Lightweight displacement spec (a=atom index, i=cartesian axis 0/1/2,
# sign in {-1, 0, +1}, ndisp in {0, 1, 2}). Mirrors ASE's Displacement
# naming so the on-disk vocabulary is recognizable.
Disp = namedtuple('Disp', ['a', 'i', 'sign', 'ndisp'])


def _disp_name(d: Disp) -> str:
    if d.sign == 0:
        return 'eq'
    return f"{d.a}{'xyz'[d.i]}{d.ndisp * ' +-'[int(d.sign)]}"


class ParallelVibrations:
    """Vibrational mode / ZPE calculator that batches displacements
    across a list of structures.

    Parameters
    ----------
    atoms_list:
        Equilibrium structures whose vibrations we want.
    batch_evaluator:
        Callable ``(atoms_list) -> (E, forces_list, stress_voigt_list)``.
        Stresses are unused here but kept in the signature so the same
        wrappers used by the parallel optimizers can be reused verbatim.
    indices:
        Per-structure list of atom indices to displace (default: all
        non-fixed). Pass a single sequence to apply the same indices to
        every structure.
    delta:
        Displacement magnitude in Å.
    nfree:
        2 (central +/-delta) or 4 (also +/-2 delta) finite-difference points.
    batch_size:
        Optional cap on the per-call batch size. ``None`` packs all
        same-composition displacements into one evaluator call.
    """

    def __init__(self,
                 atoms_list: Sequence[Atoms],
                 batch_evaluator: Callable,
                 indices: Optional[Sequence] = None,
                 delta: float = 0.01,
                 nfree: int = 2,
                 batch_size: Optional[int] = None):
        assert nfree in (2, 4)
        self.atoms_list = [a.copy() for a in atoms_list]
        self.batch_evaluator = batch_evaluator
        self.delta = delta
        self.nfree = nfree
        self.batch_size = batch_size

        self.indices_list: List[np.ndarray] = []
        for atoms in self.atoms_list:
            if indices is None:
                fixed = set()
                for constr in atoms.constraints:
                    if isinstance(constr, FixAtoms):
                        fixed.update(constr.get_indices())
                idx = [i for i in range(len(atoms)) if i not in fixed]
            else:
                idx = list(indices)
            if len(idx) != len(set(idx)):
                raise ValueError('one (or more) indices included more than once')
            self.indices_list.append(np.asarray(idx, dtype=int))

        self.cache_list: List[dict] = [{} for _ in self.atoms_list]
        self.H_list: List[Optional[np.ndarray]] = [None] * len(self.atoms_list)
        self._vibrations_list: List[Optional[VibrationsData]] = \
            [None] * len(self.atoms_list)
        self.method = 'standard'
        self.direction = 'central'

    # ------------------------------------------------------------------ disps

    def _disps_for(self, k: int) -> List[Disp]:
        out = [Disp(0, 0, 0, 0)]
        for a in self.indices_list[k]:
            for i in range(3):
                for sign in (-1, 1):
                    for ndisp in range(1, self.nfree // 2 + 1):
                        out.append(Disp(int(a), i, sign, ndisp))
        return out

    # ------------------------------------------------------------------- run

    def run(self):
        """Generate every (eq + ±delta) displacement and evaluate them.

        Structures are grouped by chemical composition; within a group we
        iterate one displacement (mode) at a time and batch across all
        structures that share that composition. This bounds the per-call
        batch size to ``n_structures`` (or ``batch_size`` if set) instead
        of ``n_structures * (1 + 6N)``, which is what most ML calculators
        actually want for memory.

        ``self.atoms_list[k]`` (already an internal copy made in ``__init__``)
        is reused as a working buffer: we shift one coordinate in place,
        evaluate, then restore it — no per-mode ``atoms.copy()``.
        """
        comp_groups: dict = defaultdict(list)
        for k in range(len(self.atoms_list)):
            atoms = self.atoms_list[k]
            sig = (len(atoms), tuple(atoms.get_chemical_symbols()))
            comp_groups[sig].append(k)

        for k_indices in comp_groups.values():
            disps = self._disps_for(k_indices[0])
            total = len(disps)
            n_struct = len(k_indices)
            bs = self.batch_size or n_struct

            for m, d in enumerate(disps):
                print(f"[ParallelVibrations] computing mode "
                      f"{m + 1}/{total} for {n_struct} atoms")
                step = d.sign * d.ndisp * self.delta  # 0.0 for eq

                for start in range(0, n_struct, bs):
                    sub_k = k_indices[start:start + bs]
                    atoms_chunk = [self.atoms_list[k] for k in sub_k]

                    if step != 0.0:
                        for atoms in atoms_chunk:
                            atoms.positions[d.a, d.i] += step
                    try:
                        _, F_list, _ = self.batch_evaluator(atoms_chunk)
                    finally:
                        if step != 0.0:
                            for atoms in atoms_chunk:
                                atoms.positions[d.a, d.i] -= step

                    for k, F in zip(sub_k, F_list):
                        self.cache_list[k][_disp_name(d)] = \
                            np.asarray(F).reshape(-1, 3)

    # ------------------------------------------------------------------ read

    def read(self, method: str = 'standard', direction: str = 'central'):
        """Assemble the per-structure Hessian from cached forces.

        Mirrors ``ase.vibrations.Vibrations.read``.
        """
        self.method = method.lower()
        self.direction = direction.lower()
        assert self.method in ('standard', 'frederiksen')
        assert self.direction in ('central', 'forward', 'backward')

        for k, atoms in enumerate(self.atoms_list):
            indices = self.indices_list[k]
            cache = self.cache_list[k]
            n = 3 * len(indices)
            H = np.empty((n, n))
            r = 0

            feq = cache['eq'] if self.direction != 'central' else None

            for a in indices:
                for i in range(3):
                    fminus = cache[_disp_name(Disp(int(a), i, -1, 1))]
                    fplus = cache[_disp_name(Disp(int(a), i, 1, 1))]
                    if self.method == 'frederiksen':
                        fminus = fminus.copy()
                        fplus = fplus.copy()
                        fminus[a] -= fminus.sum(0)
                        fplus[a] -= fplus.sum(0)
                    if self.nfree == 4:
                        fmm = cache[_disp_name(Disp(int(a), i, -1, 2))]
                        fpp = cache[_disp_name(Disp(int(a), i, 1, 2))]
                        if self.method == 'frederiksen':
                            fmm = fmm.copy()
                            fpp = fpp.copy()
                            fmm[a] -= fmm.sum(0)
                            fpp[a] -= fpp.sum(0)
                    if self.direction == 'central':
                        if self.nfree == 2:
                            H[r] = 0.5 * (fminus - fplus)[indices].ravel()
                        else:
                            H[r] = (-fmm + 8 * fminus - 8 * fplus + fpp
                                    )[indices].ravel() / 12.0
                    elif self.direction == 'forward':
                        H[r] = (feq - fplus)[indices].ravel()
                    else:
                        H[r] = (fminus - feq)[indices].ravel()
                    H[r] /= 2 * self.delta
                    r += 1

            H += H.copy().T
            self.H_list[k] = H
            self._vibrations_list[k] = VibrationsData.from_2d(
                atoms, H, indices=indices)

    # ------------------------------------------------------------- accessors

    def _ensure_read(self):
        if self.H_list[0] is None:
            self.read()

    def get_vibrations(self) -> List[VibrationsData]:
        self._ensure_read()
        return list(self._vibrations_list)

    def get_zero_point_energies(self) -> List[float]:
        return [v.get_zero_point_energy() for v in self.get_vibrations()]

    def get_energies_list(self) -> List[np.ndarray]:
        return [v.get_energies() for v in self.get_vibrations()]

    def summary(self, log=sys.stdout):
        for k, vd in enumerate(self.get_vibrations()):
            log.write(f"\n--- Structure {k} ---\n")
            for line in VibrationsData._tabulate_from_energies(vd.get_energies()):
                log.write(line + '\n')
            log.write(f"Zero-point energy: {vd.get_zero_point_energy():.4f} eV\n")


# --------------------------------------------------------------- evaluators

def make_deepmd_evaluator(calculator):
    """Wrap ``batch_calculator_deepmd`` into the
    ``(atoms_list) -> (E, F, S)`` signature expected by ParallelVibrations,
    ParallelFIRE and ParallelLBFGS."""
    def _eval(atoms_list):
        return batch_calculator_deepmd(atoms_list, calculator)
    return _eval



