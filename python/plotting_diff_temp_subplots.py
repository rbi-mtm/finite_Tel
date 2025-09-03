import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import yaml

models = [86]
temps = [0, 50, 100, 500, 800, 1000, 1050, 1100, 1150, 1200, 1300, 1500, 1800, 2000, 2300]
temps_ = [0, 800, 1000, 1150, 1200, 1500] #[0, 100, 800, 1000, 1150, 1200, 1500]

colors = cm.jet(np.linspace(0, 0.45, len(temps_))) #cm.jet(np.linspace(0, 1, len(temps_)))

temps_dft = [0, 400, 600, 800, 900, 950]

colors_dft = cm.jet(np.linspace(1, 0.55, len(temps_dft))) #cm.jet(np.linspace(0, 1, len(temps_dft)))

distances_dft = []
frequencies0_dft = []
frequencies1_dft = []
frequencies2_dft = []
frequencies3_dft = []
frequencies4_dft = []
frequencies5_dft = []
frequencies6_dft = []
frequencies7_dft = []
frequencies8_dft = []

for temp_dft in temps_dft:
    distances_temp_dft = []
    frequencies_temp0_dft = []
    frequencies_temp1_dft = []
    frequencies_temp2_dft = []
    frequencies_temp3_dft = []
    frequencies_temp4_dft = []
    frequencies_temp5_dft = []
    frequencies_temp6_dft = []
    frequencies_temp7_dft = []
    frequencies_temp8_dft = []

    if temp_dft == 0:
        with open('/storage/LUKAB_STORAGE/1H_NbSe2_vasp_phonopy_9x9x1/band.yaml', 'r') as file:
            data_dft = yaml.safe_load(file)
    else:
        with open('/storage/LUKAB_STORAGE/1H_NbSe2_vasp_phonopy_diff_temp9x9/{}/band.yaml'.format(temp_dft), 'r') as file:
            data_dft = yaml.safe_load(file)

    for i in range(data_dft['nqpoint']):
        distances_temp_dft.append(data_dft['phonon'][i]['distance'])
        frequencies_temp0_dft.append(data_dft['phonon'][i]['band'][0]['frequency'])
        frequencies_temp1_dft.append(data_dft['phonon'][i]['band'][1]['frequency'])
        frequencies_temp2_dft.append(data_dft['phonon'][i]['band'][2]['frequency'])
        frequencies_temp3_dft.append(data_dft['phonon'][i]['band'][3]['frequency'])
        frequencies_temp4_dft.append(data_dft['phonon'][i]['band'][4]['frequency'])
        frequencies_temp5_dft.append(data_dft['phonon'][i]['band'][5]['frequency'])
        frequencies_temp6_dft.append(data_dft['phonon'][i]['band'][6]['frequency'])
        frequencies_temp7_dft.append(data_dft['phonon'][i]['band'][7]['frequency'])
        frequencies_temp8_dft.append(data_dft['phonon'][i]['band'][8]['frequency'])

    distances_dft.append(np.array(distances_temp_dft))
    frequencies0_dft.append(np.array(frequencies_temp0_dft))
    frequencies1_dft.append(np.array(frequencies_temp1_dft))
    frequencies2_dft.append(np.array(frequencies_temp2_dft))
    frequencies3_dft.append(np.array(frequencies_temp3_dft))
    frequencies4_dft.append(np.array(frequencies_temp4_dft))
    frequencies5_dft.append(np.array(frequencies_temp5_dft))
    frequencies6_dft.append(np.array(frequencies_temp6_dft))
    frequencies7_dft.append(np.array(frequencies_temp7_dft))
    frequencies8_dft.append(np.array(frequencies_temp8_dft))

ticks_coord_dft = [distances_dft[0][0], distances_dft[0][100], distances_dft[0][201], distances_dft[0][302]]
ticks_labels_dft = ["$\\Gamma$", "M", "K", "$\\Gamma$"]

for model in models:
    distances = []
    frequencies0 = []
    frequencies1 = []
    frequencies2 = []
    frequencies3 = []
    frequencies4 = []
    frequencies5 = []
    frequencies6 = []
    frequencies7 = []
    frequencies8 = []

    distances_temp = np.load('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/ensemble_fermi_3/data/{}/kpoints.npy'.format(model))
    frequencies_temp0 = np.load('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/ensemble_fermi_3/data/{}/freq0.npy'.format(model))
    frequencies_temp1 = np.load('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/ensemble_fermi_3/data/{}/freq1.npy'.format(model))
    frequencies_temp2 = np.load('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/ensemble_fermi_3/data/{}/freq2.npy'.format(model))
    frequencies_temp3 = np.load('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/ensemble_fermi_3/data/{}/freq3.npy'.format(model))
    frequencies_temp4 = np.load('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/ensemble_fermi_3/data/{}/freq4.npy'.format(model))
    frequencies_temp5 = np.load('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/ensemble_fermi_3/data/{}/freq5.npy'.format(model))
    frequencies_temp6 = np.load('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/ensemble_fermi_3/data/{}/freq6.npy'.format(model))
    frequencies_temp7 = np.load('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/ensemble_fermi_3/data/{}/freq7.npy'.format(model))
    frequencies_temp8 = np.load('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/ensemble_fermi_3/data/{}/freq8.npy'.format(model))

    for i in range(len(temps)):
        if temps[i] in temps_:
            distances.append(np.array(distances_temp[i]))
            frequencies0.append(np.array(frequencies_temp0[i])) #*(1/33.356))
            frequencies1.append(np.array(frequencies_temp1[i])) #*(1/33.356))
            frequencies2.append(np.array(frequencies_temp2[i])) #*(1/33.356))
            frequencies3.append(np.array(frequencies_temp3[i])) #*(1/33.356))
            frequencies4.append(np.array(frequencies_temp4[i])) #*(1/33.356))
            frequencies5.append(np.array(frequencies_temp5[i])) #*(1/33.356))
            frequencies6.append(np.array(frequencies_temp6[i])) #*(1/33.356))
            frequencies7.append(np.array(frequencies_temp7[i])) #*(1/33.356))
            frequencies8.append(np.array(frequencies_temp8[i])) #*(1/33.356))

ticks_coord = [distances[0][0], distances[0][50], distances[0][101], distances[0][152]]
ticks_labels = ["$\\Gamma$", "M", "K", "$\\Gamma$"]

fig, ax = plt.subplots(2, 1, figsize=(3.2, 5))

for i in range(len(temps_dft)):
    ax[0].plot(distances_dft[i], frequencies0_dft[i], ls='-', color=colors_dft[i])
    ax[0].plot(distances_dft[i], frequencies1_dft[i], ls='-', color=colors_dft[i])
    ax[0].plot(distances_dft[i], frequencies2_dft[i], ls='-', color=colors_dft[i])
    ax[0].plot(distances_dft[i], frequencies3_dft[i], ls='-', color=colors_dft[i])
    ax[0].plot(distances_dft[i], frequencies4_dft[i], ls='-', color=colors_dft[i])
    ax[0].plot(distances_dft[i], frequencies5_dft[i], ls='-', color=colors_dft[i])
    ax[0].plot(distances_dft[i], frequencies6_dft[i], ls='-', color=colors_dft[i])
    ax[0].plot(distances_dft[i], frequencies7_dft[i], ls='-', color=colors_dft[i])
    ax[0].plot(distances_dft[i], frequencies8_dft[i], ls='-', color=colors_dft[i], label=r'$T_{{el}}$={} K'.format(temps_dft[i]))

for j in range(1, len(ticks_coord_dft)-1):
    ax[0].axvline(ticks_coord_dft[j], ls=':', color='black', linewidth=0.5)

ax[0].set_xticks(ticks_coord_dft)
ax[0].set_xticklabels(ticks_labels_dft, fontsize=7.5)
ax[0].tick_params(axis='y', labelsize=7.5)
y_min, y_max = ax[0].get_ylim()
ax[0].axhspan(y_min, 0, color="grey", alpha=0.15)
ax[0].text(0.025, 0.975, '(a)', transform=ax[0].transAxes, fontsize=7.5, va='top', ha='left', color='black')
ax[0].set_ylabel(r'$\omega$ [THz]', fontsize=7.5)
ax[0].set_xlim(min(distances_dft[0]), max(distances_dft[0]))
ax[0].set_ylim(-3, 10)
ax[0].axhline(0.0, ls = ':', color='black', linewidth=0.5)
ax[0].legend(loc='lower right', fontsize=7.5)

for i in range(len(temps_)):
    ax[1].plot(distances[i], frequencies0[i], ls='-', color=colors[i])
    ax[1].plot(distances[i], frequencies1[i], ls='-', color=colors[i])
    ax[1].plot(distances[i], frequencies2[i], ls='-', color=colors[i])
    ax[1].plot(distances[i], frequencies3[i], ls='-', color=colors[i])
    ax[1].plot(distances[i], frequencies4[i], ls='-', color=colors[i])
    ax[1].plot(distances[i], frequencies5[i], ls='-', color=colors[i])
    ax[1].plot(distances[i], frequencies6[i], ls='-', color=colors[i])
    ax[1].plot(distances[i], frequencies7[i], ls='-', color=colors[i])
    ax[1].plot(distances[i], frequencies8[i], ls='-', color=colors[i], label=r'$T_{{el}}$={} K'.format(temps_[i]))

for j in range(1, len(ticks_coord)-1):
    ax[1].axvline(ticks_coord[j], ls=':', color='black', linewidth=0.5)

ax[1].set_xticks(ticks_coord)
ax[1].set_xticklabels(ticks_labels, fontsize=7.5)
ax[1].tick_params(axis='y', labelsize=7.5)
y_min, y_max = ax[1].get_ylim()
ax[1].axhspan(y_min, 0, color="grey", alpha=0.15)
ax[1].text(0.025, 0.975, '(b)', transform=ax[1].transAxes, fontsize=7.5, va='top', ha='left', color='black')
ax[1].set_ylabel(r'$\omega$ [THz]', fontsize=7.5)
ax[1].set_xlim(min(distances[0]), max(distances[0]))
ax[1].set_ylim(-3, 10)
ax[1].axhline(0.0, ls = ':', color='black', linewidth=0.5)
ax[1].legend(loc='lower right', fontsize=7.5)
    
plt.tight_layout()
plt.savefig('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/ensemble_fermi_3/subplots_full.png')
plt.savefig('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/ensemble_fermi_3/subplots_full.pdf')
plt.close()
        
fig, ax = plt.subplots(2, 1, figsize=(3.2, 5))

for i in range(len(temps_dft)):
    ax[0].plot(distances_dft[i], frequencies0_dft[i], ls='-', color=colors_dft[i])
    ax[0].plot(distances_dft[i], frequencies1_dft[i], ls='-', color=colors_dft[i])
    ax[0].plot(distances_dft[i], frequencies2_dft[i], ls='-', color=colors_dft[i], label=r'$T_{{el}}$={} K'.format(temps_dft[i]))

for j in range(1, len(ticks_coord_dft)-1):
    ax[0].axvline(ticks_coord_dft[j], ls=':', color='black', linewidth=0.5)

ax[0].set_xticks(ticks_coord_dft)
ax[0].set_xticklabels(ticks_labels_dft, fontsize=7.5)
ax[0].tick_params(axis='y', labelsize=7.5)
y_min, y_max = ax[0].get_ylim()
ax[0].axhspan(y_min, 0, color="grey", alpha=0.15)
ax[0].text(0.025, 0.975, '(a)', transform=ax[0].transAxes, fontsize=7.5, va='top', ha='left', color='black')
ax[0].set_ylabel(r'$\omega$ [THz]', fontsize=7.5)
ax[0].set_xlim(min(distances_dft[0]), max(distances_dft[0]))
ax[0].set_ylim(-3, 5)
ax[0].axhline(0.0, ls = ':', color='black', linewidth=0.5)
ax[0].legend(loc='lower right', fontsize=7.5)

for i in range(len(temps_)):
    ax[1].plot(distances[i], frequencies0[i], ls='-', color=colors[i])
    ax[1].plot(distances[i], frequencies1[i], ls='-', color=colors[i])
    ax[1].plot(distances[i], frequencies2[i], ls='-', color=colors[i], label=r'$T_{{el}}$={} K'.format(temps_[i]))

for j in range(1, len(ticks_coord)-1):
    ax[1].axvline(ticks_coord[j], ls=':', color='black', linewidth=0.5)

ax[1].set_xticks(ticks_coord)
ax[1].set_xticklabels(ticks_labels, fontsize=7.5)
ax[1].tick_params(axis='y', labelsize=7.5)
y_min, y_max = ax[1].get_ylim()
ax[1].axhspan(y_min, 0, color="grey", alpha=0.15)
ax[1].text(0.025, 0.975, '(b)', transform=ax[1].transAxes, fontsize=7.5, va='top', ha='left', color='black')
ax[1].set_ylabel(r'$\omega$ [THz]', fontsize=7.5)
ax[1].set_xlim(min(distances[0]), max(distances[0]))
ax[1].set_ylim(-3, 5)
ax[1].axhline(0.0, ls = ':', color='black', linewidth=0.5)
ax[1].legend(loc='lower right', fontsize=7.5)
    
plt.tight_layout()
plt.savefig('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/ensemble_fermi_3/subplots_acoustic.png')
plt.savefig('/storage/LUKAB_STORAGE/1H_NbSe2_smearings/0.01/ensemble_fermi_3/subplots_acoustic.pdf')
plt.close()
