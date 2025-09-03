#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=1H_NbSe2_DOS_optuna_0.01_fermi_3
#SBATCH --cpus-per-task=1
#SBATCH --mem=64gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=20-0

#module load Anaconda3
#source ~/.bashrc
#export PYTHONPATH=$PYTHONPATH:"/home/lukab/anaconda3/envs/librascal/lib/python3.9/site-packages/rascal/lib"
#echo $PYTHONPATH > test2
#printenv > test2

#conda init bash
#conda activate librascal

export PATH="/home/lukab/anaconda3/envs/rascal_calc/bin:$PATH"

export MKL_CBWR="AVX2"
export I_MPI_FABRICS=shm:ofi
ulimit -s unlimited

python3 1H_NbSe2_DOS_optuna_0.01_fermi_3.py > log_luka_1H_NbSe2_DOS_optuna_0.01_fermi_3
