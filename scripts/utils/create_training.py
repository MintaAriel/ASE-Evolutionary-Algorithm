import dpdata
from deepmd.calculator import DP
from pathlib import Path
from ase.io import Trajectory, read

from ase.calculators.singlepoint import SinglePointCalculator
from ea.analysis.compare_model import CompareModel
# from pygulp.relaxation.relax import Gulp_relaxation_noadd
# from pygulp.io.read_gulp import read_results
import numpy as np



def gulp_cal(atom):
    work_dir = '/home/vito/PythonProjects/ASEProject/DeepmdTrain'

    gulp_input = (f"gradient conp conse qok c6 conp prop gfnff gwolf noauto\n"
                  f"gfnff_scale 0.8 1.343 0.727 1.0 2.859\n"
                   #f"gfnff_scale 0.5915 1.7055 0.5463 0.712 0.9907\n"
                  f"maths mrrr\n"
                  f"pressure 0 GPa"
                  )
    options = (
        "output movie cif out1.cif\n"
        "maxcycle 300\n"
        "gtol 0.00001"
    )

    cal_dir = work_dir + '/single_param_cal'
    relax = Gulp_relaxation_noadd(path=cal_dir,
                                  library=None,
                                  gulp_keywords=gulp_input,
                                  gulp_options=options)

    new = relax.use_gulp(atom)

    results = read_results(cal_dir + '/CalcFold/ginput1.got')

    return new, results


def recalculate_batch_gulp():
    # batch = read('/home/vito/PythonProjects/ASEProject/DeepmdTrain/allpbebj_vasp.traj', index=':')
    batch = read('/home/vito/PythonProjects/ASEProject/DeepmdTrain/gfnff_tune/THP4_best_2.traj', index=':')
    # calc = DP(model='/home/vito/PythonProjects/ASEProject/EA/models/dpa3-pbed3-pytorch.pth', device='gpu')
    out_dir = Path('/home/vito/PythonProjects/ASEProject/DeepmdTrain')
    out_path = out_dir / 'THP4_gfnff.traj'
    new_batch = []

    with Trajectory(out_path, mode='w') as traj_out:
        for i, atom in enumerate(batch):
            calculated, results = gulp_cal(atom)
            # Compute properties
            energy = results['energy'][0]
            grad_frac = results['gradient']
            A = atom.cell.array  # 3x3
            forces = -(grad_frac @ np.linalg.inv(A))
            virial_tensor = results['strain']
            virial = np.array([virial_tensor[0,0],
                                virial_tensor[1,1],
                                virial_tensor[2,2],
                                virial_tensor[1,2],
                                virial_tensor[0,2],
                                virial_tensor[0,1],
                            ])

            stress = -virial / atom.get_volume()
            # print(energy, stress, type(energy), type(stress))

            # 🔥 Store results explicitly
            atom.calc = SinglePointCalculator(
                atom,
                energy=energy,
                forces=forces,
                stress=stress
            )

            traj_out.write(atom)
            print(f'atom {i} done')


def recalculate_batch():
    # batch = read('/home/vito/uspex_matlab/theo_pyxtal/2THP/test_2/output.traj', index=':')
    # batch = read('/home/vito/PythonProjects/ASEProject/DeepmdTrain/allpbej_vasp.traj', index=':')
    batch = read('/home/vito/PythonProjects/ASEProject/DeepmdTrain/gfnff_tune/THP4_best_2.traj', index=':')
    calc = DP(model='/home/vito/PythonProjects/ASEProject/EA/models/dpa3-pbed3-pytorch.pth', device='gpu')
    # calc = DP(model='/home/vito/PythonProjects/ASEProject/DeepmdTrain/frozen_model.pth', device='gpu')
    # calc = DP(model='/home/vito/PythonProjects/ASEProject/EA/models/dpa3-d3_torch.pth', device='gpu')
    # calc = DP(model='/home/vito/PythonProjects/ASEProject/DeepmdTrain/frozen_model_omat.pth', device='gpu')
    out_dir = Path('/home/vito/PythonProjects/ASEProject/DeepmdTrain')

    out_path = out_dir / 'THP4_2.traj'
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
    # cmp = CompareModel(base / 'allpbebj_vasp.traj', '/home/vito/PythonProjects/ASEProject/DeepmdTrain/gfnff_tune/trial_0192/gfnff.traj')
    cmp = CompareModel(base / 'THP4_2.traj',base / 'THP4_gfnff.traj')

    cmp.plot()
    cmp.plot_forces_xyz()
    cmp.plot_virial_diag()

# recalculate_batch()
# recalculate_batch_gulp()
test_compare_model()
# import numpy as np
#
# d = read('/home/vito/PythonProjects/ASEProject/DeepmdTrain/allpbebj_gfnff_my.traj',  index=':')[100]
#
# virial = d.get_stress()
# stress = -virial/d.get_volume()
#
# print(d.get_volume())
#
# print(d.get_stress())
# print(stress)
