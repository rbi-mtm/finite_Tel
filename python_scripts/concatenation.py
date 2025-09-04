import numpy as np

xdos_fermi_0 = np.load('./xdos/xdos_0.01_fermi_0.npy')

xdos_fermi_tot = xdos_fermi_0
ldos_fermi_tot = []

np.save('./xdos_0.01_fermi_tot_all.npy', xdos_fermi_tot)

for i in range(0, 87):
    ldos = np.load('./ldos_0.01_fermi_{}.npy'.format(i))
    ldos_fermi_tot = ldos_fermi_tot + ldos.tolist()

ldos_fermi_tot = np.array(ldos_fermi_tot)

np.save('./ldos_0.01_fermi_tot_all.npy', ldos_fermi_tot)
