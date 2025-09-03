# finite_Tel
These are examples of scripts used for producing the most important results of the paper [Luka Benić et al.,  J. Chem. Theory Comput. 2025, 21, 16, 8130–8141](https://pubs.acs.org/doi/full/10.1021/acs.jctc.5c00959).


## Installation

We have installed *librascal* Python package from [this repository (finite_T_calc branch)](https://github.com/cbenmahm/librascal.git). After the installation you have to change **asemd.py** file with the one we provided in the **python_scripts** directory. Example of the path in the conda environment:

```bash
~/anaconda3/envs/rascal_calc/lib/python3.9/site-packages/rascal/models/asemd.py
```

To install conda environments one can use **cpu_environment.yml** and **gpu_environment.yml** files.<br>

For the calculations we have used CPU clusters with both Slurm and PBS workload managers, and a NVIDIA GPU cluster.<br>

For DFT calculations we have used **VASP**, version 6.4.2.


## Usage

**ADJUST ALL OF THE PATHS IN THE SCRIPTS TO YOUR FOLDER STRUCTURE!!!**

One can download necessary bigger files: **86_fermi.json**, **86_fermi-krr-weights.npy**, **xdos_0.01_fermi_tot_all.npy** and **ldos_0.01_fermi_tot_all.npy** from [Zenodo](https://zenodo.org/records/15125087).<br>

In the directory **1H_NbSe2_3x3x1** we provide an example of the files used for dataset construction using **VASP**. The given example collects 3x3x1 supercell structures.<br>

After obtaining data from **VASP** we have **OUTCAR** and **vasprun.xml** files (to format **vasprun.xml** files we use commands from the **vasprun_formatting.txt** file from the **bash_scripts** directory) and use the scripts: **wght_nrg_extractor_for_dos.py**, **dos_builder_fermi_align.py**, **concatenation.py** and **mace_sets.py** from the **python_scripts** directory to construct the datasets.<br>

We have trained an machine learning model for the electronic temperature independent part of the free energy using **MACE**. We have performed the training on the NVIDIA GPU cluster using the **.sh** script from the **bash_scripts** directory.<br>

To train the model for the electronic density of states at 0K we firstly do the hyperparameter optimization using [Optuna](https://optuna.org/) Python package, which we run using the **1H_NbSe2_DOS_optuna_0.01_fermi.py** script from the **python_scripts** directory. After the hyperparameter optimization we use full training dataset to train the models, using the **full_models.py** script from the **python_scripts** directory.<br>

We perform the harmonic phonon calculations using the **.py** script from the **python_scripts** directory. If we have also performed DFT harmonic phonon calculations with VASP we can plot both of these results using the **plotting_diff_temp_subplots.py** script from the **python_scripts** directory.<br>

To include the anharmonic effects, through the ionic temperatures, when we calculate the phonon dispersions we have used [**SSCHA**](https://sscha.eu/). We have performed the calculations where the electronic temperature is set to 0K and the ionic one changes using the **1H_NbSe2_SSCHA.py** script from the **python_scripts** directory. To perform the calculations with **SSCHA** where we also include electronic temperatures we have used the **1H_NbSe2_laser.py** script from the **python_scripts** directory.
