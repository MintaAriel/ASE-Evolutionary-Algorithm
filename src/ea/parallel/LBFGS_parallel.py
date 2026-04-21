"""Batched L-BFGS optimizer mirroring ase.optimize.lbfgs.LBFGS but with
force/energy/virial evaluations performed for the whole batch in a single
deepmd calculator call per iteration. Each structure is wrapped in a
FrechetCellFilter so atomic positions AND cell DOFs are relaxed together,
exactly like LBFGS(FrechetCellFilter(atoms)) would do for a single structure.
"""

from dataclasses import dataclass, field
from typing import Any
import numpy as np
from ase.filters import FrechetCellFilter
from ase.calculators.singlepoint import SinglePointCalculator
from ase.stress import full_3x3_to_voigt_6_stress
from .create_batch import  build_batch_deepmd
from ase.io import write
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message=r"logm result may be inaccurate.*")


@dataclass
class LBFGSState:
    atoms: Any                       # ase.Atoms (mutated in place)
    filter: Any                      # FrechetCellFilter wrapping `atoms`
    maxstep: float = 0.2
    memory: int = 100
    damping: float = 1.0
    # LBFGS history (per-structure, mirrors ASE's LBFGS internals)
    iteration: int = 0
    s: list = field(default_factory=list)
    y: list = field(default_factory=list)
    rho: list = field(default_factory=list)
    r0: Any = None
    f0: Any = None
    # bookkeeping
    converged: bool = False
    energy: float | None = None
    fmax_current: float | None = None


def _inject_results(atoms, energy, forces, virial):
    """Attach a SinglePointCalculator carrying the batched DP results so
    FrechetCellFilter.get_forces() / atoms.get_stress() return them.

    DP's virial v satisfies stress = -v / volume (ASE sign convention).
    """
    volume = atoms.get_volume()
    stress_3x3 = -np.asarray(virial).reshape(3, 3) / volume
    stress_3x3 = 0.5 * (stress_3x3 + stress_3x3.T)
    stress_voigt = full_3x3_to_voigt_6_stress(stress_3x3)
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=float(np.asarray(energy).ravel()[0]),
        forces=np.asarray(forces).reshape(-1, 3),
        stress=stress_voigt,
    )


class ParallelLBFGS:
    """L-BFGS optimizer that batches force/energy/stress evaluations across
    multiple ase.Atoms objects via a single deepmd calculator call per step.

    Per-structure L-BFGS history (s, y, rho, r0, f0, iteration) is tracked
    independently and mirrors ase.optimize.lbfgs.LBFGS.step exactly.
    """

    def __init__(self, atoms_list, calc, fmax=0.05, max_steps=200,
                 maxstep=0.2, memory=100, damping=1.0, alpha=70.0,
                 logfile='-'):
        if maxstep > 1.0:
            raise ValueError(
                f"maxstep={maxstep} is too large (must be <= 1.0 A)"
            )

        self.calc = calc
        self.fmax = fmax
        self.max_steps = max_steps
        # Initial inverse Hessian guess (diagonal): H0 = 1/alpha. Shared
        # across all structures, same as the single-structure LBFGS default.
        self.H0 = 1.0 / alpha
        self.nsteps_done = 0
        self.logfile = logfile

        self.states = []
        for at in atoms_list:
            at_copy = at.copy()
            flt = FrechetCellFilter(at_copy)
            self.states.append(
                LBFGSState(atoms=at_copy, filter=flt,
                           maxstep=maxstep, memory=memory, damping=damping)
            )

    def _log(self, msg):
        if self.logfile == '-':
            print(msg)
        elif self.logfile is not None:
            with open(self.logfile, 'a') as fh:
                fh.write(msg + '\n')

    def _determine_step(self, dr, maxstep):
        """Rescale step so the largest per-row (per-atom / per-cell-row)
        displacement does not exceed `maxstep`, preserving the search
        direction. Mirrors ase.optimize.lbfgs.LBFGS.determine_step.
        """
        longest = np.linalg.norm(dr.reshape(-1, 3), axis=1).max()
        if longest >= maxstep:
            dr = dr * (maxstep / longest)
        return dr

    def step(self):
        active_idx = [i for i, s in enumerate(self.states) if not s.converged]
        if not active_idx:
            return True

        active_atoms = [self.states[i].atoms for i in active_idx]

        # Single batched GPU call for all active structures
        coords, cells, types = build_batch_deepmd(active_atoms, self.calc.type_dict)
        E, F, V = self.calc.dp.eval(coords, cells, types)[:3]

        for j, i in enumerate(active_idx):
            st = self.states[i]

            # Inject batched (E, F, V) so FrechetCellFilter can build the
            # combined (natoms+3, 3) force vector for positions + cell DOFs.
            _inject_results(st.atoms, E[j], F[j], V[j])

            force_vec = st.filter.get_forces()          # (natoms+3, 3)
            fnorm_max = float(np.linalg.norm(force_vec, axis=1).max())
            st.fmax_current = fnorm_max
            st.energy = float(np.asarray(E[j]).ravel()[0])

            if fnorm_max < self.fmax:
                st.converged = True
                continue

            forces = force_vec.ravel()                  # 1D, length = ndofs
            pos = st.filter.get_positions().ravel()

            # --- update L-BFGS history (mirrors LBFGS.update) -----------
            if st.iteration > 0:
                s0 = pos - st.r0
                y0 = st.f0 - forces                     # gradient difference
                st.s.append(s0)
                st.y.append(y0)
                st.rho.append(1.0 / np.dot(y0, s0))
            if st.iteration > st.memory:
                st.s.pop(0)
                st.y.pop(0)
                st.rho.pop(0)

            # --- two-loop recursion ------------------------------------
            loopmax = int(min(st.memory, st.iteration))
            a = np.empty(loopmax, dtype=np.float64)
            q = -forces                                 # = gradient
            for k in range(loopmax - 1, -1, -1):
                a[k] = st.rho[k] * np.dot(st.s[k], q)
                q = q - a[k] * st.y[k]
            z = self.H0 * q
            for k in range(loopmax):
                b = st.rho[k] * np.dot(st.y[k], z)
                z = z + st.s[k] * (a[k] - b)
            p = -z                                       # descent direction

            # --- apply step with maxstep rescale + damping -------------
            dr = self._determine_step(p, st.maxstep) * st.damping
            st.filter.set_positions((pos + dr).reshape(-1, 3))

            # remember for next iteration
            st.r0 = pos
            st.f0 = forces
            st.iteration += 1

        return False

    def run(self, out_dir):

        for step in range(self.max_steps):
            if step % 50 == 0:
                out_directory = Path(out_dir) / 'output.traj'
                write(out_directory, self.get_atoms())

            self.nsteps_done = step + 1
            done = self.step()


            n_conv = sum(s.converged for s in self.states)
            fmax_str = ', '.join(
                f"{s.fmax_current:.4f}" if s.fmax_current is not None else "N/A"
                for s in self.states
            )
            e_str = ', '.join(
                f"{s.energy:.4f}" if s.energy is not None else "N/A"
                for s in self.states
            )
            self._log(
                f"[ParallelLBFGS] step {self.nsteps_done:4d}  "
                f"converged {n_conv}/{len(self.states)}  "
                f"fmax=[{fmax_str}]  E=[{e_str}]"
            )
            if done:
                self._log(
                    f"[ParallelLBFGS] all converged in {self.nsteps_done} steps"
                )
                out_directory = Path(out_dir) / 'output.traj'
                write(out_directory, self.get_atoms())
                return True

        self._log(
            f"[ParallelLBFGS] reached max_steps={self.max_steps} "
            f"without full convergence"
        )


        return False

    def get_atoms(self):
        for s in self.states:
            s.atoms.calc = SinglePointCalculator(
                                                s.atoms,
                                                energy=s.energy)
        return [s.atoms for s in self.states]



