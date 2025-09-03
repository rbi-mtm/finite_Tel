import numpy as np

xdos_fermi_0 = np.load('1H_NbSe2_data_for_dos_alignments_0.01/xdos/xdos_0.01_fermi_0.npy')

xdos_fermi_tot = xdos_fermi_0
ldos_fermi_tot = []

np.save('1H_NbSe2_data_for_dos_alignments_0.01/xdos_0.01_fermi_tot_all.npy', xdos_fermi_tot)

for i in range(0, 87):
    ldos = np.load('1H_NbSe2_data_for_dos_alignments_0.01/ldos/ldos_0.01_fermi_{}.npy'.format(i))
    ldos_fermi_tot = ldos_fermi_tot + ldos.tolist()

ldos_fermi_tot = np.array(ldos_fermi_tot)

np.save('1H_NbSe2_data_for_dos_alignments_0.01/ldos_0.01_fermi_tot_all.npy', ldos_fermi_tot)
