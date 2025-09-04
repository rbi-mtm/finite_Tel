import numpy as np
import scipy
import matplotlib.pyplot as plt
from copy import deepcopy
import ase.units
from ase.io import read, write
import xml.etree.ElementTree as ET
from collections import OrderedDict
from utils import *
from ase.build.supercells import make_supercell

dx = 0.00025 #eV

kb = ase.units.kB

emin_global, emax_global = None, None

for i in range(0, 36):
    fermi_energies = np.load('./1H_NbSe2_fermi_energies_{}.npy'.format(i))
    weights_ = np.load('./1H_NbSe2_wght_dos_{}.npy'.format(i))
    energies = np.load('./1H_NbSe2_nrg_dos_{}.npy'.format(i))
    frames = read('./1H_NbSe2_frames_{}.xyz'.format(i), index=':')

    weights = []
    for j in weights_:
        weights.append(j)

    for j in range(len(energies)):
        energies[j] = energies[j] - fermi_energies[j]
 
    energies_mins = np.array([x.min() for x in energies])
    energies_maxs = np.array([x.max() for x in energies])
    emin, emax = np.min(energies_mins), np.max(energies_maxs)

    if i==0:
        emin_global = emin
        emax_global = emax

    natoms = np.zeros(len(frames), int)
    for j in range(len(frames)):
        natoms[j] = len(frames[j])

    xdos, ldos = build_dos(0.01,
	                   energies, 
		           dx, 
		           emin_global,
		           emax_global, 
		           weights=weights,
		           natoms=natoms)

    np.save('./xdos_0.01_fermi_{}.npy'.format(i), xdos)
    np.save('./ldos_0.01_fermi_{}.npy'.format(i), ldos)
