#!/bin/bash
#SBATCH --partition=cm
#SBATCH --job-name=wght_nrg_extractor_for_dos
#SBATCH --cpus-per-task=1
#SBATCH --mem=64gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10-0

export PATH="~/anaconda3/envs/rascal_calc/bin:$PATH"

export MKL_CBWR="AVX2"
export I_MPI_FABRICS=shm:ofi
ulimit -s unlimited

python3 wght_nrg_extractor_for_dos.py > log_wght_nrg_extractor_for_dos
