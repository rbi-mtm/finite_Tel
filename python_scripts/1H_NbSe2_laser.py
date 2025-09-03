from ase.optimize import BFGS
from ase.constraints import ExpCellFilter
from ase.spacegroup.symmetrize import FixSymmetry
from ase.io import *
from ase.phonons import Phonons
import cellconstructor as CC
import cellconstructor.Structure
import cellconstructor.Phonons
import ase
import cellconstructor.ForceTensor
import sscha, sscha.Ensemble, sscha.Relax, sscha.SchaMinimizer, sscha.Utilities
import spglib
from datetime import datetime
from ase import Atom, Atoms
import fnmatch
import os
import numpy as np
import matplotlib.pyplot as plt
import ase.units
from ase.build.supercells import make_supercell
from ase.spacegroup.symmetrize import FixSymmetry
from ase.constraints import ExpCellFilter
import faulthandler
from rascal.models.asemd import ASEFiniteTCalculator
from mace.calculators import MACECalculator
from ase.calculators.fd import FiniteDifferenceCalculator
import json
import glob
from copy import deepcopy
from ase.constraints import FixedLine

np.random.seed(0)

T_el = 400                                  #electronic temperature
temp = 60                                   #initial ionic temperature
temps = [70, 80, 90, 100, 110, 120, 130]    #ionic temperatures     

lowest_hessian_mode = []
lowest_sscha_mode = []

configurations = 250
max_population = 100

sobol = False
sobol_scramble = False

PATH = "GMKG"
N_POINTS = 1000

SPECIAL_POINTS = {"G": [0, 0, 0], "M": [1/2, 0, 0], "K": [1/3, 1/3, 0]}

atoms = ase.io.read("./c2db-12553.json")

atoms_ = deepcopy(atoms)

M = [[9,0,0], [0,9,0], [0,0,1]]
atoms9x9 = make_supercell(atoms, M)
ase.io.write('./atoms9x9.xyz', atoms9x9, format='extxyz')

constraints9x9 = [FixedLine(i, [0, 0, 1]) for i in range(len(atoms9x9))]
atoms9x9.set_constraint(constraints9x9)

atoms = atoms9x9

E0_calc = MACECalculator(model_paths='./1H_NbSe2_model_6_run-123_swa.model', device='cuda')

DOS_calc = ASEFiniteTCalculator(model_json='./86_fermi.json', is_periodic=True,
                                xdos="./xdos_0.01_fermi_tot.npy", temperature=T_el,
                                ref_temperature=10,
                                structure_template='./atoms9x9.xyz',
                                choice = "energy_entr_0",
                                nelectrons=1863, ref_nelectrons=1863, contribution="all")
                                
total_calculator = ase.calculators.mixing.SumCalculator(calcs=[E0_calc, DOS_calc])

atoms.calc = total_calculator

dyn_ = BFGS(atoms)
dyn_.run(fmax=1e-4)

atoms2 = Atoms()
atoms2.cell = atoms_.cell
atoms2.pbc = atoms_.pbc
atoms2 += atoms[0]
atoms2 += atoms[1]
atoms2 += atoms[2]
    
ph = Phonons(atoms2, total_calculator, supercell=(9, 9, 1), delta=0.01)
ph.run()

ph.read(acoustic=True)
ph.clean()
    
x = CC.Phonons.get_dyn_from_ase_phonons(ase_ph = ph, adjust_qstar = True)
    
x.Symmetrize()

x.ForcePositiveDefinite()
    
x.save_qe("./dyn_{}K_".format(temp))
    
nqirr = len(fnmatch.filter(os.listdir('./'), 'dyn_{}K_*'.format(temp)))
    
dyn = CC.Phonons.Phonons("./dyn_{}K_".format(temp), nqirr = nqirr)

ensemble = sscha.Ensemble.Ensemble(dyn, T0 = temp, supercell = dyn.GetSupercell())
    
minim = sscha.SchaMinimizer.SSCHA_Minimizer(ensemble)
minim.min_step_struc = 0.05
minim.min_step_dyn = 0.05
minim.kong_liu_ratio = 0.5
minim.meaningful_factor = 0.00001
minim.enforce_sum_rule = True

relax = sscha.Relax.SSCHA(minim, ase_calculator = total_calculator, N_configs = configurations, max_pop = max_population)

ioinfo = sscha.Utilities.IOInfo()
ioinfo.SetupSaving("./minim_info_{}K".format(temp))
relax.setup_custom_functions(custom_function_post = ioinfo.CFP_SaveAll)

relax.relax(sobol = sobol, sobol_scramble = sobol_scramble)
    
relax.minim.dyn.save_qe("./final_dyn_{}K_".format(temp))

symm = spglib.get_spacegroup(relax.minim.dyn.structure.get_ase_atoms(), 1e-5)
print('Current SG = ', symm,' at T=', temp)

ensemble = sscha.Ensemble.Ensemble(relax.minim.dyn, T0 = temp, supercell = dyn.GetSupercell())
ensemble.generate(configurations*4, sobol = sobol, sobol_scramble = sobol_scramble)
    
ensemble.get_energy_forces(total_calculator, compute_stress = False)

ensemble.update_weights(relax.minim.dyn, temp)
    
dyn_hessian = ensemble.get_free_energy_hessian(include_v4 = False)

dyn_hessian.save_qe("./hessian_{}K_".format(temp))

w_sscha, pols_sscha = relax.minim.dyn.DiagonalizeSupercell()

superstructure = relax.minim.dyn.structure.generate_supercell(relax.minim.dyn.GetSupercell())

acoustic_modes = CC.Methods.get_translations(pols_sscha, superstructure.get_masses_array())
w_sscha = w_sscha[~acoustic_modes]
lowest_sscha_mode.append(np.min(w_sscha) * CC.Units.RY_TO_CM)

w_hessian, pols_hessian = dyn_hessian.DiagonalizeSupercell()

acoustic_modes = CC.Methods.get_translations(pols_hessian, superstructure.get_masses_array())
w_hessian = w_hessian[~acoustic_modes]
lowest_hessian_mode.append(np.min(w_hessian) * CC.Units.RY_TO_CM)
    
hessian_dyn = CC.Phonons.Phonons("./hessian_{}K_".format(temp), nqirr)

qpath, data = CC.Methods.get_bandpath(hessian_dyn.structure.unit_cell, PATH, SPECIAL_POINTS, N_POINTS)
    
xaxis, xticks, xlabels = data
    
hessian_dispersion = CC.ForceTensor.get_phonons_in_qpath(hessian_dyn, qpath)

nmodes = dyn_hessian.structure.N_atoms * 3

np.save('./xticks_{}K.npy'.format(temp), np.array(xticks))
np.save('./xlabels_{}K.npy'.format(temp), np.array(xlabels))
np.save('./kpoints_{}K.npy'.format(temp), np.array(xaxis))
np.save('./freq0_{}K.npy'.format(temp), np.array(hessian_dispersion[:,0]))
np.save('./freq1_{}K.npy'.format(temp), np.array(hessian_dispersion[:,1]))
np.save('./freq2_{}K.npy'.format(temp), np.array(hessian_dispersion[:,2]))

plt.figure(dpi = 150)
ax = plt.gca()

for i in range(nmodes):
    ax.plot(xaxis, hessian_dispersion[:,i], color='r')
        
for x in xticks:
    ax.axvline(x, 0, 1, color = "k", ls = '--', lw = 0.4)
    ax.axhline(0, 0, 1, color = 'k', ls = ':', lw = 0.4)

ax.set_xticks(xticks)
ax.set_xticklabels(xlabels)

ax.set_xlabel("Q path")
ax.set_ylabel("Phonons [cm-1]")

ax.legend()
    
plt.tight_layout()
plt.savefig("./acoustic_phonons_{}K.png".format(temp))
plt.close()

dyn_pop_files = glob.glob(os.path.join('./', 'dyn_pop*'))

for file in dyn_pop_files:
    try:
        os.remove(file)
        print(f"Removed: {file}")
    except FileNotFoundError:
        print(f"File not found: {file}")

temp_old = temp

for temp in temps:
    dyn = CC.Phonons.Phonons("./final_dyn_{}K_".format(temp_old), nqirr = nqirr)

    ensemble = sscha.Ensemble.Ensemble(dyn, T0 = temp, supercell = dyn.GetSupercell())
    
    minim = sscha.SchaMinimizer.SSCHA_Minimizer(ensemble)
    minim.min_step_struc = 0.05
    minim.min_step_dyn = 0.05
    minim.kong_liu_ratio = 0.5
    minim.meaningful_factor = 0.00001
    minim.enforce_sum_rule = True

    relax = sscha.Relax.SSCHA(minim, ase_calculator = total_calculator, N_configs = configurations, max_pop = max_population)

    ioinfo = sscha.Utilities.IOInfo()
    ioinfo.SetupSaving("./minim_info_{}K".format(temp))
    relax.setup_custom_functions(custom_function_post = ioinfo.CFP_SaveAll)

    relax.relax(sobol = sobol, sobol_scramble = sobol_scramble)
    
    relax.minim.dyn.save_qe("./final_dyn_{}K_".format(temp))

    symm = spglib.get_spacegroup(relax.minim.dyn.structure.get_ase_atoms(), 1e-5)
    print('Current SG = ', symm,' at T={}K'.format(temp))

    ensemble = sscha.Ensemble.Ensemble(relax.minim.dyn, T0 = temp, supercell = dyn.GetSupercell())
    ensemble.generate(configurations*4, sobol = sobol, sobol_scramble = sobol_scramble)
    
    ensemble.get_energy_forces(total_calculator, compute_stress = False)

    ensemble.update_weights(relax.minim.dyn, temp)
    
    dyn_hessian = ensemble.get_free_energy_hessian(include_v4 = False) #True

    dyn_hessian.save_qe("./hessian_{}K_".format(temp))

    w_sscha, pols_sscha = relax.minim.dyn.DiagonalizeSupercell()

    superstructure = relax.minim.dyn.structure.generate_supercell(relax.minim.dyn.GetSupercell())

    acoustic_modes = CC.Methods.get_translations(pols_sscha, superstructure.get_masses_array())
    w_sscha = w_sscha[~acoustic_modes]
    lowest_sscha_mode.append(np.min(w_sscha) * CC.Units.RY_TO_CM) #CC.Units.RY_TO_EV

    w_hessian, pols_hessian = dyn_hessian.DiagonalizeSupercell()

    acoustic_modes = CC.Methods.get_translations(pols_hessian, superstructure.get_masses_array())
    w_hessian = w_hessian[~acoustic_modes]
    lowest_hessian_mode.append(np.min(w_hessian) * CC.Units.RY_TO_CM) #CC.Units.RY_TO_EV
    
    hessian_dyn = CC.Phonons.Phonons("./hessian_{}K_".format(temp), nqirr)

    qpath, data = CC.Methods.get_bandpath(hessian_dyn.structure.unit_cell, PATH, SPECIAL_POINTS, N_POINTS)
    
    xaxis, xticks, xlabels = data
    
    hessian_dispersion = CC.ForceTensor.get_phonons_in_qpath(hessian_dyn, qpath)

    nmodes = dyn_hessian.structure.N_atoms * 3

    np.save('./xticks_{}K.npy'.format(temp), np.array(xticks))
    np.save('./xlabels_{}K.npy'.format(temp), np.array(xlabels))
    np.save('./kpoints_{}K.npy'.format(temp), np.array(xaxis))
    np.save('./freq0_{}K.npy'.format(temp), np.array(hessian_dispersion[:,0]))
    np.save('./freq1_{}K.npy'.format(temp), np.array(hessian_dispersion[:,1]))
    np.save('./freq2_{}K.npy'.format(temp), np.array(hessian_dispersion[:,2]))

    plt.figure(dpi = 150)
    ax = plt.gca()

    for i in range(nmodes):
        ax.plot(xaxis, hessian_dispersion[:,i], color='r')
        
    for x in xticks:
        ax.axvline(x, 0, 1, color = "k", ls = '--', lw = 0.4)
        ax.axhline(0, 0, 1, color = 'k', ls = ':', lw = 0.4)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)

    ax.set_xlabel("Q path")
    ax.set_ylabel("Phonons [cm-1]")

    ax.legend()
    
    plt.tight_layout()
    plt.savefig("./acoustic_phonons_{}K.png".format(temp))
    plt.close()
    
    dyn_pop_files = glob.glob(os.path.join('./', 'dyn_pop*'))
    
    if temp != temps[-1]:
        for file in dyn_pop_files:
            try:
                os.remove(file)
                print(f"Removed: {file}")
            except FileNotFoundError:
                print(f"File not found: {file}")
    
    temp_old = temp
    
temps_ = [60, 70, 80, 90, 100, 110, 120, 130]

freq_data = np.zeros((len(temps_), 3))
freq_data[:, 0] = temps_
freq_data[:, 1] = lowest_sscha_mode
freq_data[:, 2] = lowest_hessian_mode

np.savetxt("./{}_hessian_vs_temperature.dat".format(configurations), freq_data, header = "T [K]; SSCHA mode [cm-1]; Free energy hessian [cm-1]")
