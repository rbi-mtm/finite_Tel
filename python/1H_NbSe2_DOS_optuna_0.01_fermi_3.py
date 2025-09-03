import numpy as np
import scipy
import matplotlib.pyplot as plt

from copy import deepcopy
import ase.units
from scipy.optimize import curve_fit
from sklearn.utils.validation import check_array

import ase.io as ase_io
import ase.io.vasp
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

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from scipy.integrate import trapezoid

from utils import *
from rascal.representations import SphericalInvariants
from rascal.models import KRR

import optuna
import joblib
import math

from collections import Counter

smearing = 0.01 #eV
dx = 0.00025 #eV

kb = ase.units.kB

frames_tot = ase_io.read("/storage/LUKAB_STORAGE/1H_NbSe2_data_for_dos_alignments_0.01/dos_data/1H_NbSe2_frames_tot_ML_all.xyz", ":")

natoms = np.zeros(len(frames_tot), int)
for j in range(len(frames_tot)):
    natoms[j] = len(frames_tot[j])

for x in frames_tot:
    x.wrap(eps=1e-10)		
    
xdos = None
ldos = None

xdos = np.load("/storage/LUKAB_STORAGE/1H_NbSe2_data_for_dos_alignments_0.01/xdos_0.01_fermi_tot_all.npy")
ldos = np.load("/storage/LUKAB_STORAGE/1H_NbSe2_data_for_dos_alignments_0.01/ldos_0.01_fermi_tot_all.npy")

np.random.seed(10)
ntot = len(frames_tot)

itrain = np.arange(ntot) 
np.random.shuffle(itrain)
itest = itrain[int(0.8*ntot):]
itrain = itrain[:int(0.8*ntot)]

frames_train, frames_test = [], []

for i in itrain:
    frames_train.append(frames_tot[i])

for i in itest:
    frames_test.append(frames_tot[i])

np.save("/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/1H_NbSe2_itrain_DOS_optuna_0.01_fermi_3.npy", itrain)
np.save("/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/1H_NbSe2_itest_DOS_optuna_0.01_fermi_3.npy", itest)

mean_dos_per_atom = np.mean((ldos.T / natoms).T, axis=0)
np.save("/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/1H_NbSe2_mean_dos_per_atom_optuna_0.01_fermi_3.npy", mean_dos_per_atom)

mean_dos = np.zeros((ntot, ldos.shape[1]))
for j in range(ntot):
    mean_dos[j] = natoms[j] * mean_dos_per_atom

train_dos = ldos - mean_dos

n_trials = 100

def objective(trial, xdos, ldos, mean_dos, frames_tot, frames_train, train_dos, itrain, itest):
    np.random.seed(trial.number)
    itrain_temp = np.arange(len(itrain))
    np.random.shuffle(itrain_temp)
    ivalidation_temp = itrain_temp[int(0.8*len(itrain_temp)):]
    itrain_temp = itrain_temp[:int(0.8*len(itrain_temp))]

    np.save("/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/indices_temp_fermi_3/1H_NbSe2_itrain_temp_DOS_optuna_0.01_fermi_3_model_{}.npy".format(trial.number), itrain_temp)
    np.save("/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/indices_temp_fermi_3/1H_NbSe2_ivalidation_temp_DOS_optuna_0.01_fermi_3_model_{}.npy".format(trial.number), ivalidation_temp)

    frames_train_temp = []

    for i in itrain_temp:
        frames_train_temp.append(frames_train[i])

    species_counts = []
    for frame in frames_train_temp:
        species = frame.get_chemical_symbols()
        species_count = Counter(species)
        species_counts.append(species_count)

    Nb_count, Se_count = 0, 0

    for i in species_counts:
        Nb_count += int(i['Nb'])
        Se_count += int(i['Se'])

    interaction_cutoff = trial.suggest_float('interaction_cutoff', 4.0, 10.0, step=0.1)

    scale_perc = trial.suggest_float('scale_perc', 0.4, 0.9, step=0.1)

    scale = scale_perc * interaction_cutoff

    hypers = dict(soap_type="PowerSpectrum",
              interaction_cutoff = interaction_cutoff, 
              max_radial = trial.suggest_int('max_radial', 8, 16, step=1), 
              max_angular = trial.suggest_int('max_angular', 8, 16, step=1), 
              gaussian_sigma_constant = trial.suggest_float('gaussian_sigma_constant', 0.1, 1.5, step=0.1),
              gaussian_sigma_type = "Constant",
              cutoff_smooth_width = trial.suggest_float('cutoff_smooth_width', 0.5, 1.5, step=0.1),
              normalize = True,
              radial_basis = "GTO",
              cutoff_function_type = 'RadialScaling',
              cutoff_function_parameters = dict(rate = trial.suggest_float('rate', 0.5, 2.5, step=0.1),
                                                scale = scale,
                                                exponent = trial.suggest_int('exponent', 1, 6, step=1)),
              expansion_by_species_method = 'structure wise',
              global_species = [34, 41]
              )

    soap = SphericalInvariants(**hypers)

    train_managers = soap.transform(frames_train_temp)

    n_sparse_perc = trial.suggest_categorical('n_sparse_perc', [0.005, 0.0075, 0.01, 0.0125])

    n_sparse = {34: int(n_sparse_perc*Se_count), 41: int(n_sparse_perc*Nb_count)} #{34:152, 41:112}

    sparse = FPSFilter(soap, n_sparse, act_on='sample per species')
    X_sparse = sparse.select_and_filter(train_managers)

    hypers_grad = deepcopy(hypers)
    hypers_grad["compute_gradients"] = True
    soap_grad = SphericalInvariants(**hypers_grad)

    zeta = trial.suggest_int('zeta', 2, 4, step = 1)

    kernel = Kernel(soap_grad, 
                    name='GAP', 
                    zeta=zeta, 
                    target_type='Structure', 
                    kernel_type='Sparse')

    kNM = []

    for j in range(len(frames_tot)):
        k = compute_KNM([frames_tot[j]], X_sparse, kernel, soap_grad)
        kNM.append(k[0])

    kNM = np.array(kNM)

    np.save('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/kNM_fermi_3/{}_fermi.npy'.format(trial.number), kNM)

    feat_ref = X_sparse.get_features()
    kMM = (feat_ref @ feat_ref.T)**zeta

    eigval, eigvec = scipy.linalg.eigh(kMM)
    eigval = eigval[::-1]
    eigvec = eigvec[:, ::-1]

    plt.semilogy(eigval/eigval[0])
    plt.xlabel("Selected feature index")
    plt.ylabel("Hausdorff distance")
    plt.savefig('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/optuna_reg1_hausdorff_3/fermi/{}_hausdorff.png'.format(trial.number))
    plt.close()

    threshold_perc = trial.suggest_categorical('threshold_perc', [1/3, 1/2, 2/3, 3/4])

    exponent = math.floor(math.log10(abs(min(eigval/eigval[0]))))

    threshold = 10**(threshold_perc*exponent)
    
    nrkhs = ((eigval / eigval[0]) > threshold).sum()

    transfMat = np.dot(eigvec[:, :nrkhs], np.diag(1./np.sqrt(eigval[:nrkhs])))
    
    reg_arr = np.logspace(-10, 5, 16)

    cv = 5
    
    kfold = KFold(n_splits = cv, shuffle = False)

    errors = []
    for regularization in reg_arr:
        temp_err = 0.
        for train, test in kfold.split(itrain_temp):
            w = get_regression_weights(train_dos[itrain][itrain_temp][train], 
                                       kMM=kMM,
                                       transfMat=transfMat,
                                       regularization1=regularization, 
                                       kNM=kNM[itrain][itrain_temp][train])
            target_pred = kNM @ w
            temp_err += get_rmse(target_pred[itrain][itrain_temp][test], 
                                 train_dos[itrain][itrain_temp][test], 
                                 xdos,
                                 perc=True)
        errors.append(temp_err/cv)
    errors = np.asarray(errors)

    plt.loglog(reg_arr, errors[:], "o-")
    plt.xlabel("Regularization")
    plt.ylabel("%RMSE")
    plt.savefig('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/optuna_reg1_hausdorff_3/fermi/{}_reg1.png'.format(trial.number))
    plt.close()

    regularization1 = reg_arr[errors.argmin()]

    weights = get_regression_weights(train_dos[itrain][itrain_temp],
                                     kMM=kMM,
                                     transfMat=transfMat,
                                     regularization1=regularization1,
                                     regularization2=None,
                                     kNM=kNM[itrain][itrain_temp],
                                     gradients=False,
                                     nn=None)

    ldos_pred = kNM @ weights
    ldos_pred += mean_dos

    ldos_pred = np.array(ldos_pred)

    np.save('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/ldos_fermi_3/ldos_{}.npy'.format(trial.number), ldos_pred)

    scorer_train_flattened = get_score(ldos_pred[itrain][itrain_temp].flatten(), ldos[itrain][itrain_temp].flatten())
    scorer_validation_flattened = get_score(ldos_pred[itrain][ivalidation_temp].flatten(), ldos[itrain][ivalidation_temp].flatten())
    scorer_test_flattened = get_score(ldos_pred[itest].flatten(), ldos[itest].flatten())

    with open('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/statistics_fermi_3.txt', 'a') as f:
        f.write('model_{}_statistics: '.format(trial.number)+'\n')
        f.write('scale: '+str(scale)+'\n')
        f.write('threshold: '+str(threshold)+'\n')
        f.write('regularization1: '+str(regularization1)+'\n')
        f.write('n_sparse_34: '+str(n_sparse[34])+'\n')
        f.write('n_sparse_41: '+str(n_sparse[41])+'\n')
        f.write('%RMSE_train: '+str(get_rmse(ldos_pred[itrain][itrain_temp], ldos[itrain][itrain_temp], xdos=xdos, perc=True))+'%'+'\n')
        f.write('%RMSE_validation: '+str(get_rmse(ldos_pred[itrain][ivalidation_temp], ldos[itrain][ivalidation_temp], xdos=xdos, perc=True))+'%'+'\n')
        f.write('%RMSE_test: '+str(get_rmse(ldos_pred[itest], ldos[itest], xdos=xdos, perc=True))+'%'+'\n')
        f.write('%RMSE_validation/%RMSE_train: '+str(get_rmse(ldos_pred[itrain][ivalidation_temp], ldos[itrain][ivalidation_temp], xdos=xdos, perc=True)/get_rmse(ldos_pred[itrain][itrain_temp], ldos[itrain][itrain_temp], xdos=xdos, perc=True))+'\n')
        f.write('%RMSE_test/%RMSE_train: '+str(get_rmse(ldos_pred[itest], ldos[itest], xdos=xdos, perc=True)/get_rmse(ldos_pred[itrain][itrain_temp], ldos[itrain][itrain_temp], xdos=xdos, perc=True))+'\n')
        f.write('%RMSE_validation/%RMSE_test: '+str(get_rmse(ldos_pred[itrain][ivalidation_temp], ldos[itrain][ivalidation_temp], xdos=xdos, perc=True)/get_rmse(ldos_pred[itest], ldos[itest], xdos=xdos, perc=True))+'\n')
        f.write('r2_train_flattened: '+str(scorer_train_flattened['R2'])+'\n')
        f.write('r2_validation_flattened: '+str(scorer_validation_flattened['R2'])+'\n')
        f.write('r2_test_flattened: '+str(scorer_test_flattened['R2'])+'\n')
        f.write('\n')

    model = KRR(weights, 
                kernel, 
                X_sparse,
                self_contributions =  {34: mean_dos_per_atom, 41: mean_dos_per_atom},
                description = "model description")

    dump_obj("/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/models_fermi_3/{}_fermi.json".format(trial.number), model)
    
    error = get_rmse(ldos_pred[itrain][ivalidation_temp], ldos[itrain][ivalidation_temp], xdos, perc=True)
    
    return error

sampler = optuna.samplers.TPESampler(n_startup_trials = int(0.1 * n_trials))
study = optuna.create_study(sampler = sampler, direction = "minimize")

func = lambda trial: objective(trial, xdos = xdos, ldos = ldos, mean_dos = mean_dos, frames_tot = frames_tot, frames_train = frames_train, train_dos = train_dos, itrain = itrain, itest = itest)

study.optimize(func, n_trials = n_trials)
joblib.dump(study, "/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/1H_NbSe2_DOS_optuna_0.01_fermi_3.pkl")

print("Best params:")
for key, value in study.best_params.items():
    print(f"\t{key}: {value}")

print("Optimized RMSE:", study.best_value)
