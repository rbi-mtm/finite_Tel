# finite_Tel
These are examples of scripts used for producing the results of the paper [Luka Benić et al.,  J. Chem. Theory Comput. 2025, 21, 16, 8130–8141](https://pubs.acs.org/doi/full/10.1021/acs.jctc.5c00959).


## Installation

We have installed *librascal* Python package from [this repository (finite_T_calc branch)](https://github.com/cbenmahm/librascal.git). After the installation you have to change **asemd.py** file with the one we provided in the **python** directory. Example of the path in the conda environment:

```bash
~/anaconda3/envs/rascal_calc/lib/python3.9/site-packages/rascal/models/asemd.py
```

To install conda environments one can use **cpu_environment.yml** and **gpu_environment.yml** files.<br>

For the calculations we have used CPU clusters with both Slurm and PBS workload managers, and a GPU cluster with NVIDIA GPU units.<br>

For DFT calculations we have used **VASP**, version 6.4.2.


## Usage

**ADJUST ALL OF THE PATHS IN THE SCRIPTS TO YOUR FOLDER STRUCTURE!!!**

One can download necessary bigger files from [Zenodo](https://zenodo.org/records/15125087): **86_fermi.json**, **86_fermi-krr-weights.npy**, **xdos_0.01_fermi_tot_all.npy** and **ldos_0.01_fermi_tot_all.npy**.

In the directory **1H_NbSe2_3x3x1** we provide an example of the files used for dataset construction using **VASP**. The given example collects 3x3x1 supercell structures.
