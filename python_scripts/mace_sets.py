from ase.io import read, write
import numpy as np

frames = read("./1H_NbSe2_frames_tot_MACE.xyz", ":")

np.random.seed(10)
ntot = len(frames)
ntrain = int(0.8 * ntot)

itrain = np.arange(ntot)
np.random.shuffle(itrain)
itest = itrain[ntrain:]
itrain = itrain[:ntrain]

frames_train, frames_test = [], []

for i in itrain:
    frames_train.append(frames[i])

for i in itest:
    frames_test.append(frames[i])

write('./1H_NbSe2_frames_train_MACE.xyz', frames_train, format="extxyz")
write('./1H_NbSe2_frames_test_MACE.xyz', frames_test, format="extxyz")
