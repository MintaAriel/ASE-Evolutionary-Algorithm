import dpdata
from pathlib import Path
from ase.io import Trajectory, read
from deepmd.calculator import DP
from ase.calculators.singlepoint import SinglePointCalculator
from ea.analysis.compare_model import CompareModel


def recalculate_batch():
    # batch = read('/home/vito/uspex_matlab/theo_pyxtal/2THP/test_2/output.traj', index=':')
    batch = read('/home/vito/PythonProjects/ASEProject/DeepmdTrain/unrelaxed.trajectory', index=':100')
    # calc = DP(model='/home/vito/PythonProjects/ASEProject/EA/models/dpa3-pbed3-pytorch.pth', device='gpu')
    calc = DP(model='/home/vito/PythonProjects/ASEProject/DeepmdTrain/frozen_model.pth', device='gpu')
    # calc = DP(model='/home/vito/PythonProjects/ASEProject/EA/models/dpa3-d3_torch.pth', device='gpu')
    # calc = DP(model='/home/vito/PythonProjects/ASEProject/DeepmdTrain/frozen_model_omat.pth', device='gpu')
    out_dir = Path('/home/vito/PythonProjects/ASEProject/DeepmdTrain')

    out_path = out_dir / 'omol25_unrel.traj'
    new_batch = []

    with Trajectory(out_path, mode='w') as traj_out:
        for i, atom in enumerate(batch):
            atom.calc = calc

            # Compute properties
            energy = atom.get_potential_energy()
            forces = atom.get_forces()
            stress = atom.get_stress()

            # 🔥 Store results explicitly
            atom.calc = SinglePointCalculator(
                atom,
                energy=energy,
                forces=forces,
                stress=stress
            )

            traj_out.write(atom)
            print(f'atom {i} done')



#
# trial = batch[:5]
#
# # data = dpdata.LabeledSystem('/home/vito/VASP/parallel_vasp_trial/vasp_0/OUTCAR', fmt='vasp/outcar')
# # data = dpdata.LabeledSystem(trial, fmt='ase/structure')

def create_dpdata():
    batch = read('/home/vito/PythonProjects/ASEProject/DeepmdTrain/results.traj', index=':')
    dpdata_system = None
    #
    for atoms in batch[110:]:
        frame = dpdata.LabeledSystem(atoms, fmt="ase/structure")

        if dpdata_system is None:
            dpdata_system = frame
        else:
            dpdata_system.append(frame)

    out_dir = Path('/home/vito/PythonProjects/ASEProject/DeepmdTrain')
    print(dpdata_system)
    dpdata_system.to_deepmd_npy(out_dir / 'validate')


def test_compare_model():
    base = Path('/home/vito/PythonProjects/ASEProject/DeepmdTrain')
    cmp = CompareModel(base / 'pbed3_unrel.traj', base / 'omat24_unrel.traj')

    cmp.plot()
    cmp.plot_forces_xyz()
    cmp.plot_virial_diag()


# recalculate_batch()
test_compare_model()
