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
from matplotlib.ticker import LogFormatterSciNotation

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

ntot = len(frames_tot)

itrain = np.load("/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/1H_NbSe2_itrain_DOS_optuna_0.01_fermi_3.npy")
itest = np.load("/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/1H_NbSe2_itest_DOS_optuna_0.01_fermi_3.npy")

frames_train, frames_test = [], []

for i in itrain:
    frames_train.append(frames_tot[i])

for i in itest:
    frames_test.append(frames_tot[i])

species_counts = []

for frame in frames_train:
    species = frame.get_chemical_symbols()
    species_count = Counter(species)
    species_counts.append(species_count)

Nb_count, Se_count = 0, 0

for i in species_counts:
    Nb_count += int(i['Nb'])
    Se_count += int(i['Se'])

mean_dos_per_atom = np.load("/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/1H_NbSe2_mean_dos_per_atom_optuna_0.01_fermi_3.npy")

mean_dos = np.zeros((ntot, ldos.shape[1]))
for j in range(ntot):
    mean_dos[j] = natoms[j] * mean_dos_per_atom

train_dos = ldos - mean_dos

n_sparse_percs, threshold_percs, zetas = [], [], []

lines_ = []

with open('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/1H_NbSe2_DOS_optuna_0.01_fermi_3-387252.err','r') as f:
    lines = f.readlines()
    for line in lines:
        splt = line.split()
        if 'Trial' in splt:
            lines_.append(splt)
            
for i in lines_:
    for j in range(len(i)):
        if i[j] == "'zeta':":
            zetas.append(int(i[j+1][:-1]))
        elif i[j] == "'n_sparse_perc':":
            n_sparse_percs.append(float(i[j+1][:-1]))
        elif i[j] == "'threshold_perc':":
            threshold_percs.append(float(i[j+1][:-2]))

for mdl in [86]: #range(0,100):
    model = load_obj("/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/models_fermi_3/{}_fermi.json".format(mdl))
    hlp = model.get_representation_calculator()
    
    hypers = dict(soap_type="PowerSpectrum",
                  interaction_cutoff = hlp.hypers['cutoff_function']['cutoff']['value'], 
                  max_radial = hlp.hypers['max_radial'], 
                  max_angular = hlp.hypers['max_angular'], 
                  gaussian_sigma_constant = hlp.hypers['gaussian_density']['gaussian_sigma']['value'],
                  gaussian_sigma_type = "Constant",
                  cutoff_smooth_width = hlp.hypers['cutoff_function']['smooth_width']['value'],
                  normalize = True,
                  radial_basis = "GTO",
                  cutoff_function_type = 'RadialScaling',
                  cutoff_function_parameters = dict(rate = hlp.hypers['cutoff_function']['rate']['value'],
                                                    scale = hlp.hypers['cutoff_function']['scale']['value'],
                                                    exponent = hlp.hypers['cutoff_function']['exponent']['value']),
                  expansion_by_species_method = 'structure wise',
                  global_species = [34, 41])

    soap = SphericalInvariants(**hypers)

    train_managers = soap.transform(frames_train)

    n_sparse_perc = n_sparse_percs[mdl]

    n_sparse = {34: int(n_sparse_perc*Se_count), 41: int(n_sparse_perc*Nb_count)} #{34:152, 41:112}

    sparse = FPSFilter(soap, n_sparse, act_on='sample per species')
    X_sparse = sparse.select_and_filter(train_managers)

    hypers_grad = deepcopy(hypers)
    hypers_grad["compute_gradients"] = True
    soap_grad = SphericalInvariants(**hypers_grad)

    zeta = zetas[mdl]

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

    np.save('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/full_models_fermi_3/kNM/{}_fermi_2.npy'.format(mdl), kNM)

    feat_ref = X_sparse.get_features()
    kMM = (feat_ref @ feat_ref.T)**zeta

    eigval, eigvec = scipy.linalg.eigh(kMM)
    eigval = eigval[::-1]
    eigvec = eigvec[:, ::-1]

    plt.figure(figsize=(3.2, 2.5))
    plt.semilogy(eigval/eigval[0])
    plt.xlabel("Selected feature index", fontsize=7.5)
    plt.ylabel("Hausdorff distance", fontsize=7.5)
    plt.xticks(fontsize=7.5)
    plt.yticks(fontsize=7.5)
    plt.tight_layout()
    plt.savefig('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/full_models_fermi_3/reg1_hausdorff/{}_hausdorff_2.png'.format(mdl))
    plt.savefig('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/full_models_fermi_3/reg1_hausdorff/{}_hausdorff_2.pdf'.format(mdl))
    plt.close()

    threshold_perc = threshold_percs[mdl]

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
        for train, test in kfold.split(itrain):
            w = get_regression_weights(train_dos[itrain][train], 
                                       kMM=kMM,
                                       transfMat=transfMat,
                                       regularization1=regularization, 
                                       kNM=kNM[itrain][train])
            target_pred = kNM @ w
            temp_err += get_rmse(target_pred[itrain][test], 
                                 train_dos[itrain][test], 
                                 xdos,
                                 perc=True)
        errors.append(temp_err/cv)
    errors = np.asarray(errors)

    plt.figure(figsize=(3.2, 2.5))
    plt.loglog(reg_arr, errors[:], "o-", ms=4)
    plt.xlabel("Regularization", fontsize=7.5)
    plt.ylabel("%RMSE", fontsize=7.5)

    # Get current axis
    ax = plt.gca()

    # Ensure consistent font size for ticks
    ax.tick_params(axis='both', which='both', labelsize=7.5)

    # Use LogFormatterSciNotation to keep scientific notation uniform
    ax.xaxis.set_major_formatter(LogFormatterSciNotation())
    ax.yaxis.set_major_formatter(LogFormatterSciNotation())

    #plt.xticks(fontsize=7.5)
    #plt.yticks(fontsize=7.5)

    plt.tight_layout()
    plt.savefig('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/full_models_fermi_3/reg1_hausdorff/{}_reg1_2.png'.format(mdl))
    plt.savefig('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/full_models_fermi_3/reg1_hausdorff/{}_reg1_2.pdf'.format(mdl))
    plt.close()

    regularization1 = reg_arr[errors.argmin()]

    weights = get_regression_weights(train_dos[itrain],
                                     kMM=kMM,
                                     transfMat=transfMat,
                                     regularization1=regularization1,
                                     regularization2=None,
                                     kNM=kNM[itrain],
                                     gradients=False,
                                     nn=None)

    ldos_pred = kNM @ weights
    ldos_pred += mean_dos

    ldos_pred = np.array(ldos_pred)

    np.save('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/full_models_fermi_3/ldos/ldos_{}_2.npy'.format(mdl), ldos_pred)

    scorer_train_flattened = get_score(ldos_pred[itrain].flatten(), ldos[itrain].flatten())
    scorer_test_flattened = get_score(ldos_pred[itest].flatten(), ldos[itest].flatten())

    with open('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/full_models_fermi_3/statistics_2.txt', 'a') as f:
        f.write('model_{}_statistics: '.format(mdl)+'\n')
        f.write('threshold: '+str(threshold)+'\n')
        f.write('regularization1: '+str(regularization1)+'\n')
        f.write('n_sparse_34: '+str(n_sparse[34])+'\n')
        f.write('n_sparse_41: '+str(n_sparse[41])+'\n')
        f.write('%RMSE_train: '+str(get_rmse(ldos_pred[itrain], ldos[itrain], xdos=xdos, perc=True))+'%'+'\n')
        f.write('%RMSE_test: '+str(get_rmse(ldos_pred[itest], ldos[itest], xdos=xdos, perc=True))+'%'+'\n')
        f.write('%RMSE_test/%RMSE_train: '+str(get_rmse(ldos_pred[itest], ldos[itest], xdos=xdos, perc=True)/get_rmse(ldos_pred[itrain], ldos[itrain], xdos=xdos, perc=True))+'\n')
        f.write('r2_train_flattened: '+str(scorer_train_flattened['R2'])+'\n')
        f.write('r2_test_flattened: '+str(scorer_test_flattened['R2'])+'\n')
        f.write('\n')

    model = KRR(weights, 
                kernel, 
                X_sparse,
                self_contributions =  {34: mean_dos_per_atom, 41: mean_dos_per_atom},
                description = "model description")

    dump_obj("/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/full_models_fermi_3/models/{}_fermi_2.json".format(mdl), model)












