from .create_batch import build_batch_deepmd
from dataclasses import dataclass
from typing import IO, Any
from ase.filters import FrechetCellFilter
from ase.calculators.singlepoint import SinglePointCalculator
from ase.stress import full_3x3_to_voigt_6_stress
from collections.abc import Callable
import numpy as np



@dataclass
class FIREState:
    atoms: Any                       # original ase.Atoms (will be mutated in-place)
    filter: Any                      # FrechetCellFilter wrapping `atoms`
    dt: float = 0.1
    maxstep: float = 0.2
    dtmax: float = 1.0
    Nmin: int = 5
    finc: float = 1.1
    fdec: float = 0.5
    astart: float = 0.1
    fa: float = 0.99
    a: float = 0.1
    Nsteps: int = 0
    vel: Any = None                  # 1D array over (natoms+3)*3 DOFs
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
    # symmetrize for robustness then fold to Voigt-6
    stress_3x3 = 0.5 * (stress_3x3 + stress_3x3.T)
    stress_voigt = full_3x3_to_voigt_6_stress(stress_3x3)
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=float(np.asarray(energy).ravel()[0]),
        forces=np.asarray(forces).reshape(-1, 3),
        stress=stress_voigt,
    )


class ParallelFIRE:
    """FIRE optimizer that batches force/energy/stress evaluations across
    multiple ase.Atoms objects via a single deepmd calculator call per step.

    Each structure is wrapped in a FrechetCellFilter so both atomic positions
    AND cell degrees of freedom are relaxed together, exactly like
        FIRE(FrechetCellFilter(atoms))
    would do for a single structure. Per-structure FIRE state (vel, dt, a,
    Nsteps) is tracked independently and mirrors ase.optimize.fire.FIRE.step.
    """

    def __init__(self, atoms_list, calc, fmax=0.05, max_steps=200,
                 dt=0.1, maxstep=0.03, dtmax=1.0, Nmin=5,
                 finc=1.1, fdec=0.5, astart=0.1, fa=0.99, a=0.1,
                 logfile='-'):
        self.states = []
        for at in atoms_list:
            at_copy = at.copy()
            flt = FrechetCellFilter(at_copy)
            self.states.append(
                FIREState(atoms=at_copy, filter=flt,
                          dt=dt, maxstep=maxstep, dtmax=dtmax,
                          Nmin=Nmin, finc=finc, fdec=fdec, astart=astart,
                          fa=fa, a=a)
            )
        self.calc = calc
        self.fmax = fmax
        self.max_steps = max_steps
        self.nsteps_done = 0
        self.logfile = logfile

    def _log(self, msg):
        if self.logfile == '-':
            print(msg)
        elif self.logfile is not None:
            with open(self.logfile, 'a') as fh:
                fh.write(msg + '\n')

    def step(self):
        active_idx = [i for i, s in enumerate(self.states) if not s.converged]
        if not active_idx:
            return True

        active_atoms = [self.states[i].atoms for i in active_idx]

        # Single batched GPU call for all active structures
        coords, cells, types = build_batch_deepmd(active_atoms, self.calc.type_dict)
        E, F, V = self.calc.dp.eval(coords, cells, types)[:3]
        # E: (B, 1) or (B,); F: (B, N, 3) or (B, N*3); V: (B, 9) or (B, 3, 3)

        for j, i in enumerate(active_idx):
            s = self.states[i]

            # Inject batched results into a SinglePointCalculator so the
            # FrechetCellFilter can compute the combined (natoms+3, 3)
            # force vector (atomic forces + cell log-deformation gradient).
            _inject_results(s.atoms, E[j], F[j], V[j])

            force_vec = s.filter.get_forces()        # (natoms+3, 3)
            fnorm_max = float(np.linalg.norm(force_vec, axis=1).max())
            s.fmax_current = fnorm_max
            s.energy = float(np.asarray(E[j]).ravel()[0])

            if fnorm_max < self.fmax:
                s.converged = True
                continue

            # ASE FIRE convention: gradient = -(-forces) = +forces (flat)
            gradient = force_vec.ravel()

            if s.vel is None:
                s.vel = np.zeros_like(gradient)
            else:
                vf = np.vdot(gradient, s.vel)
                grad2 = np.vdot(gradient, gradient)
                if vf > 0.0:
                    s.vel = ((1.0 - s.a) * s.vel
                             + s.a * gradient / np.sqrt(grad2)
                             * np.sqrt(np.vdot(s.vel, s.vel)))
                    if s.Nsteps > s.Nmin:
                        s.dt = min(s.dt * s.finc, s.dtmax)
                        s.a *= s.fa
                    s.Nsteps += 1
                else:
                    s.vel[:] = 0.0
                    s.a = s.astart
                    s.dt *= s.fdec
                    s.Nsteps = 0

            s.vel = s.vel + s.dt * gradient
            dr = s.dt * s.vel
            normdr = float(np.sqrt(np.vdot(dr, dr)))
            if normdr > s.maxstep:
                dr = s.maxstep * dr / normdr

            # Let the filter update atomic positions AND cell together (UnitCellFilter)
            x = s.filter.get_positions().ravel()
            # print('before',s.filter.atoms.cell)
            s.filter.set_positions((x + dr).reshape(-1, 3))
            # print('after',s.filter.atoms.cell)
            # s.atoms = s.filter.atoms

        return False

    def run(self):
        for step in range(self.max_steps):
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
                f"[ParallelFIRE] step {self.nsteps_done:4d}  "
                f"converged {n_conv}/{len(self.states)}  "
                f"fmax=[{fmax_str}]  E=[{e_str}]"
            )
            if done:
                self._log(
                    f"[ParallelFIRE] all converged in {self.nsteps_done} steps"
                )
                return True

        self._log(
            f"[ParallelFIRE] reached max_steps={self.max_steps} "
            f"without full convergence"
        )
        return False

    def get_atoms(self):
        return [s.atoms for s in self.states]

