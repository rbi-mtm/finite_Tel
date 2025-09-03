#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=1H_NbSe2_DOS_optuna_0.01_fermi
#SBATCH --cpus-per-task=1
#SBATCH --mem=64gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=20-0

export PATH="~/anaconda3/envs/rascal_calc/bin:$PATH"

export MKL_CBWR="AVX2"
export I_MPI_FABRICS=shm:ofi
ulimit -s unlimited

python3 1H_NbSe2_DOS_optuna_0.01_fermi.py > log_1H_NbSe2_DOS_optuna_0.01_fermi
