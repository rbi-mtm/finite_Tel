import numpy as np
import scipy
import matplotlib.pyplot as plt
from tqdm.notebook import tnrange, tqdm

from copy import deepcopy
import ase.units
from scipy.optimize import curve_fit
from sklearn.utils.validation import check_array

import ase.io as ase_io
import ase.io.vasp
import ase.calculators.mixing
import xml.etree.ElementTree as ET
from collections import OrderedDict

from rascal.neighbourlist.structure_manager import (
        mask_center_atoms_by_species, mask_center_atoms_by_id)

from rascal.utils import from_dict, to_dict, CURFilter, dump_obj, load_obj, get_score, print_score, FPSFilter
from rascal.models import Kernel, sparse_points, train_gap_model, KRR, compute_KNM

from rascal.representations import SphericalExpansion, SphericalInvariants
from rascal.utils import (get_radial_basis_covariance, get_radial_basis_pca, 
                          get_radial_basis_projections, get_optimal_radial_basis_hypers )
from rascal.utils import radial_basis

from sklearn.model_selection import KFold

from utils import *
from rascal.utils import load_obj, dump_obj
from rascal.representations import SphericalInvariants
from rascal.models import KRR

from rascal.models.genericmd import FiniteTCalculator
from rascal.models.genericmd import GenericMDCalculator
from rascal.models.asemd import ASEMLCalculator
from rascal.models.asemd import ASEFiniteTCalculator
from rascal.neighbourlist.structure_manager import AtomsList, unpack_ase

import optuna
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances

import joblib

from ase.phonons import Phonons
from ase.optimize import BFGS, FIRE
from ase.build.supercells import make_supercell

from ase.io import write

from ase.spacegroup.symmetrize import FixSymmetry
from phonopy.structure.atoms import PhonopyAtoms
from ase.constraints import ExpCellFilter

from ase import Atoms
import phonopy as ph
import faulthandler
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

from mace.calculators import MACECalculator

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.io.trajectory import Trajectory

from phonopy.structure.cells import get_primitive, get_supercell

from ase.build.tools import sort

from ase.constraints import FixedLine

from ase.calculators.fd import FiniteDifferenceCalculator

import os

def get_crystal(structure):
    cell = PhonopyAtoms(symbols=structure.get_chemical_symbols(),
                       cell=structure.cell[:],
                       scaled_positions=structure.get_scaled_positions())
    return cell

def phonopy_pre_process(cell, supercell_matrix):
    smat = supercell_matrix
    phonon = ph.Phonopy(cell, smat,
                        primitive_matrix=[[1, 0.0, 0.0],
                                         [0.0, 1, 0.0],
                                         [0.0, 0.0, 1]],)
    phonon.generate_displacements(distance=0.015)
    return phonon

def run_calc(calc, phonon):
    supercells = phonon.supercells_with_displacements
    set_of_forces = []
    for scell in supercells:
        cell = Atoms(symbols=scell.get_chemical_symbols(),
                     scaled_positions=scell.get_scaled_positions(),
                     cell=scell.get_cell(),
                     pbc=True)
        cell.set_calculator(calc)
        forces = cell.get_forces()
        drift_force = forces.sum(axis=0)
        for force in forces:
            force -= drift_force / forces.shape[0]
        set_of_forces.append(forces)
    return set_of_forces

def phonopy_post_process(phonon, interval):
    phonon.set_mesh([20, 20, 1])
    phonon.run_total_dos(freq_min=-20,freq_max=115,freq_pitch=(115+20)/interval,use_tetrahedron_method=False,sigma=0.3)
    omege=[]
    dosevi=[]
    for omega, dos in np.array(phonon.get_total_DOS()).T:
        omege.append(omega)
        dosevi.append(dos)
    return omege,dosevi

E0_calc = MACECalculator(model_paths='./1H_NbSe2_model_6_run-123_swa.model', device='cpu')

temps = [0, 50, 100, 500, 800, 1000, 1050, 1100, 1150, 1200, 1300, 1500, 1800, 2000, 2300]

for model in [86]:
    os.makedirs('1H_NbSe2_smearings/0.01/ensemble_fermi_3/phonons/{}'.format(model), exist_ok=True)
    os.makedirs('1H_NbSe2_smearings/0.01/ensemble_fermi_3/data/{}'.format(model), exist_ok=True)

    k_points = []
    branch_frequencies0, branch_frequencies1, branch_frequencies2, branch_frequencies3, branch_frequencies4, branch_frequencies5, branch_frequencies6, branch_frequencies7, branch_frequencies8 = [], [], [], [], [], [], [], [], []    

    for temp in temps:
        M = [[9,0,0], [0,9,0], [0,0,1]]
        atoms = None
        atoms = ase_io.read("./c2db-12553.json")
 
        atoms.pbc = True

        atoms_ = deepcopy(atoms)

        atoms9x9 = make_supercell(atoms, M) 

        ase_io.write('./atoms9x9_fermi.xyz', atoms9x9, format='extxyz')

        constraints9x9 = [FixedLine(i, [0, 0, 1]) for i in range(len(atoms9x9))]

        atoms9x9.set_constraint(constraints9x9)

        atoms = atoms9x9

        if temp == 0:
            total_calculator = E0_calc

        else:
            DOS_calc = ASEFiniteTCalculator(model_json='./{}_fermi.json'.format(model), is_periodic=True,
                                            xdos="./xdos_0.01_fermi_tot.npy", temperature=temp,
                                            ref_temperature=10,
                                            structure_template="./atoms9x9_fermi.xyz",
                                            choice="energy_entr_0",
                                            nelectrons=1863, ref_nelectrons=1863, contribution="all")
    
            total_calculator = ase.calculators.mixing.SumCalculator(calcs=[E0_calc, DOS_calc])

        atoms.calc = total_calculator

        dyn = BFGS(atoms)
        dyn.run(fmax=1e-4)

        atoms.calc = None

        atoms2 = Atoms()
        atoms2.cell = atoms_.cell
        atoms2.pbc = atoms_.pbc
        atoms2 += atoms[0]
        atoms2 += atoms[1]
        atoms2 += atoms[2]

        cell = get_crystal(atoms2)

        phonon = phonopy_pre_process(cell, supercell_matrix=[[9,0,0], [0,9,0], [0,0,1]])
        set_of_forces = run_calc(total_calculator, phonon)
        phonon.produce_force_constants(forces=set_of_forces)
    
        path = [[[0, 0, 0], [0.5, 0, 0], [1/3, 1/3, 0], [0, 0 ,0]]]
        labels = ["$\\Gamma$", "M", "K", "$\\Gamma$"]
        qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=51)
        phonon.run_band_structure(qpoints, path_connections=connections, labels=labels, with_eigenvectors=False, is_band_connection=False,)
        phonon.run_mesh([20, 20, 1])
        phonon.run_total_dos()

        phonon.plot_band_structure_and_dos().savefig('./9x9_{}K.png'.format(model,temp))
        plt.close()

        qpoints, distances, frequencies, eigenvectors = phonon.get_band_structure() 

        k_points_temp = []
        branch_frequencies0_temp, branch_frequencies1_temp, branch_frequencies2_temp, branch_frequencies3_temp, branch_frequencies4_temp, branch_frequencies5_temp, branch_frequencies6_temp, branch_frequencies7_temp, branch_frequencies8_temp = [], [], [], [], [], [], [], [], []

        for dist in distances:
            k_points_temp.extend(dist)

        for i in frequencies:
            for j in i:
                branch_frequencies0_temp.append(j[0])
                branch_frequencies1_temp.append(j[1])
                branch_frequencies2_temp.append(j[2])
                branch_frequencies3_temp.append(j[3])
                branch_frequencies4_temp.append(j[4])
                branch_frequencies5_temp.append(j[5])
                branch_frequencies6_temp.append(j[6])
                branch_frequencies7_temp.append(j[7])
                branch_frequencies8_temp.append(j[8])

        k_points.append(np.array(k_points_temp))
        branch_frequencies0.append(np.array(branch_frequencies0_temp))
        branch_frequencies1.append(np.array(branch_frequencies1_temp))
        branch_frequencies2.append(np.array(branch_frequencies2_temp))
        branch_frequencies3.append(np.array(branch_frequencies3_temp))
        branch_frequencies4.append(np.array(branch_frequencies4_temp))
        branch_frequencies5.append(np.array(branch_frequencies5_temp))
        branch_frequencies6.append(np.array(branch_frequencies6_temp))
        branch_frequencies7.append(np.array(branch_frequencies7_temp))
        branch_frequencies8.append(np.array(branch_frequencies8_temp))

    np.save('./{}/kpoints.npy'.format(model), np.array(k_points))
    np.save('./{}/freq0.npy'.format(model), np.array(branch_frequencies0))
    np.save('./{}/freq1.npy'.format(model), np.array(branch_frequencies1))
    np.save('./{}/freq2.npy'.format(model), np.array(branch_frequencies2))
    np.save('./{}/freq3.npy'.format(model), np.array(branch_frequencies3))
    np.save('./{}/freq4.npy'.format(model), np.array(branch_frequencies4))
    np.save('./{}/freq5.npy'.format(model), np.array(branch_frequencies5))
    np.save('./{}/freq6.npy'.format(model), np.array(branch_frequencies6))
    np.save('./{}/freq7.npy'.format(model), np.array(branch_frequencies7))
    np.save('./{}/freq8.npy'.format(model), np.array(branch_frequencies8))
