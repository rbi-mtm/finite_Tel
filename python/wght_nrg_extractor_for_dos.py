import numpy as np
import scipy
import matplotlib.pyplot as plt
from copy import deepcopy
import ase.units
from ase.io import read, write
import ase.io.vasp
import xml.etree.ElementTree as ET
from collections import OrderedDict
from utils import *
import os
import copy
import json
import shutil

for number in range(0,87):
    fermi_energies = []
    weights_dos = []
    energies_dos = []
    frames = []
    n_frames, nkpoints = None, None
    with open('1H_NbSe2_data_for_dos_alignments_0.01/OUTCAR_{}'.format(number),'r') as f:
        lines = f.readlines()
        counter = []
        for i, line in enumerate(lines):
            splt = line.split()
            if 'Fermi' in splt and 'energy:' in splt:
                counter.append(float(splt[2]))
                fermi_energies.append(float(splt[2]))
            if 'irreducible' in splt and 'k-points:' in splt:
                nkpoints = int(splt[1])

    n_frames = len(counter)

    tree = ET.iterparse(r'1H_NbSe2_data_for_dos_alignments_0.01/vasprun_{}.xml'.format(number), events=['start', 'end'])

    kpt_weights = None

    for event, elem in tree:
        if event == 'end':
            if elem.tag == 'kpoints':
                kpt_weights = elem.findall('varray[@name="weights"]/v')
                kpt_weights = [np.float64(val.text) for val in kpt_weights]

    for i in range(n_frames):
        weights_dos.append(np.array(kpt_weights))

    nrgs = []

    with open('1H_NbSe2_data_for_dos_alignments_0.01/OUTCAR_{}'.format(number),'r') as scfout:
        scf = scfout.readlines()
        for i, line in enumerate(scf):
            if line.startswith('  band No.'):
                cc = []
                j = 1
                while (scf[i+j] != '\n'):
                    cc.append(scf[i+j].split()[1])
                    j += 1
                nrgs.append([np.float64(n) for n in cc if n != '\n'])

    nrgs = np.array(nrgs)

    for i in range(0, len(nrgs), nkpoints):
        j = i
        hlp = []
        while j != i + nkpoints:
            hlp.append(nrgs[j])
            j += 1
        hlp = np.array(hlp)
        energies_dos.append(hlp)

    frames.extend(read("1H_NbSe2_data_for_dos_alignments_0.01/vasprun_{}.xml".format(number), index = ':'))

    np.save('1H_NbSe2_data_for_dos_alignments_0.01/dos_data/1H_NbSe2_wght_dos_{}.npy'.format(number), np.array(weights_dos))
    np.save('1H_NbSe2_data_for_dos_alignments_0.01/dos_data/1H_NbSe2_nrg_dos_{}.npy'.format(number), np.array(energies_dos))
    np.save('1H_NbSe2_data_for_dos_alignments_0.01/dos_data/1H_NbSe2_fermi_energies_{}.npy'.format(number), np.array(fermi_energies))
    write('1H_NbSe2_data_for_dos_alignments_0.01/dos_data/1H_NbSe2_frames_{}.xyz'.format(number), frames, format="extxyz")

frames_tot_mace = read('1H_NbSe2_data_for_dos_alignments_0.01/dos_data/1H_NbSe2_frames_0.xyz', index=':') #[]

for i in range(1,87):
    fr = read('1H_NbSe2_data_for_dos_alignments_0.01/dos_data/1H_NbSe2_frames_{}.xyz'.format(i), index=':')
    frames_tot_mace = frames_tot_mace + fr

write('1H_NbSe2_data_for_dos_alignments_0.01/dos_data/1H_NbSe2_frames_tot_MACE.xyz', frames_tot_mace, format="extxyz")

frames_tot_ml = read('1H_NbSe2_data_for_dos_alignments_0.01/dos_data/1H_NbSe2_frames_0.xyz', index=':')

for i in range(1,87):
    fr = read('1H_NbSe2_data_for_dos_alignments_0.01/dos_data/1H_NbSe2_frames_{}.xyz'.format(i), index=':')
    frames_tot_ml = frames_tot_ml + fr

write('1H_NbSe2_data_for_dos_alignments_0.01/dos_data/1H_NbSe2_frames_tot_ML.xyz', frames_tot_ml, format="extxyz")
