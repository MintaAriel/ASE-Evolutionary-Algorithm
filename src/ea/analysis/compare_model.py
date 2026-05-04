import numpy as np
import matplotlib.pyplot as plt
from ase.io import read


class CompareModel:
    def __init__(self, traj1, traj2):
        self.atoms1 = traj1 if isinstance(traj1, list) else read(traj1, index=':')
        self.atoms2 = traj2 if isinstance(traj2, list) else read(traj2, index=':')

        if len(self.atoms1) != len(self.atoms2):
            raise ValueError(
                f'Trajectories have different lengths: {len(self.atoms1)} vs {len(self.atoms2)}'
            )

    @staticmethod
    def _virial(atoms):
        return -atoms.get_stress() * atoms.get_volume()

    def _collect(self):
        e1, e2 = [], []
        f1, f2 = [], []
        v1, v2 = [], []

        for a1, a2 in zip(self.atoms1, self.atoms2):
            n = len(a1)
            e1.append(a1.get_potential_energy() / n)
            e2.append(a2.get_potential_energy() / n)
            f1.append(a1.get_forces().ravel())
            f2.append(a2.get_forces().ravel())
            v1.append(self._virial(a1) / n)
            v2.append(self._virial(a2) / n)

        return (
            np.array(e1), np.array(e2),
            np.concatenate(f1), np.concatenate(f2),
            np.concatenate(v1), np.concatenate(v2),
        )

    @staticmethod
    def _scatter(ax, x, y, label, units):
        rmse = float(np.sqrt(np.mean((x - y) ** 2)))
        mae = float(np.mean(np.abs(x - y)))
        ax.scatter(x, y, s=10, alpha=0.5, edgecolor='none')
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], 'r-', lw=1, label='y = x')
        ax.set_xlabel(f'{label} (reference) [{units}]')
        ax.set_ylabel(f'{label} (model) [{units}]')
        ax.set_title(f'{label}  RMSE={rmse:.4g}  MAE={mae:.4g}')
        ax.legend()
        ax.set_aspect('equal', adjustable='datalim')

    def plot(self, save_path=None, show=True):
        e1, e2, f1, f2, v1, v2 = self._collect()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        self._scatter(axes[0], e1, e2, 'Energy / atom', 'eV/atom')
        self._scatter(axes[1], f1, f2, 'Forces', 'eV/Å')
        self._scatter(axes[2], v1, v2, 'Virial / atom', 'eV/atom')

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200)
        if show:
            plt.show()
        return fig

    def _collect_force_components(self):
        fx1, fy1, fz1 = [], [], []
        fx2, fy2, fz2 = [], [], []

        for a1, a2 in zip(self.atoms1, self.atoms2):
            f1 = a1.get_forces()
            f2 = a2.get_forces()
            fx1.append(f1[:, 0]); fy1.append(f1[:, 1]); fz1.append(f1[:, 2])
            fx2.append(f2[:, 0]); fy2.append(f2[:, 1]); fz2.append(f2[:, 2])

        return (
            np.concatenate(fx1), np.concatenate(fy1), np.concatenate(fz1),
            np.concatenate(fx2), np.concatenate(fy2), np.concatenate(fz2),
        )

    def plot_forces_xyz(self, save_path=None, show=True):
        fx1, fy1, fz1, fx2, fy2, fz2 = self._collect_force_components()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        self._scatter(axes[0], fx1, fx2, 'Fx', 'eV/Å')
        self._scatter(axes[1], fy1, fy2, 'Fy', 'eV/Å')
        self._scatter(axes[2], fz1, fz2, 'Fz', 'eV/Å')

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200)
        if show:
            plt.show()
        return fig

    def _collect_virial_diag(self):
        vxx1, vyy1, vzz1 = [], [], []
        vxx2, vyy2, vzz2 = [], [], []

        for a1, a2 in zip(self.atoms1, self.atoms2):
            n = len(a1)
            v1 = self._virial(a1) / n
            v2 = self._virial(a2) / n
            vxx1.append(v1[0]); vyy1.append(v1[1]); vzz1.append(v1[2])
            vxx2.append(v2[0]); vyy2.append(v2[1]); vzz2.append(v2[2])

        return (
            np.array(vxx1), np.array(vyy1), np.array(vzz1),
            np.array(vxx2), np.array(vyy2), np.array(vzz2),
        )

    def plot_virial_diag(self, save_path=None, show=True):
        vxx1, vyy1, vzz1, vxx2, vyy2, vzz2 = self._collect_virial_diag()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        self._scatter(axes[0], vxx1, vxx2, 'Vxx / atom', 'eV/atom')
        self._scatter(axes[1], vyy1, vyy2, 'Vyy / atom', 'eV/atom')
        self._scatter(axes[2], vzz1, vzz2, 'Vzz / atom', 'eV/atom')

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200)
        if show:
            plt.show()
        return fig
