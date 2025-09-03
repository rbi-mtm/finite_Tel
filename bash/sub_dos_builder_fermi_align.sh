#!/bin/bash
#SBATCH --partition=cm
#SBATCH --job-name=dos_builder_fermi
#SBATCH --cpus-per-task=1
#SBATCH --mem=32gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10-0

export PATH="~/anaconda3/envs/rascal_calc/bin:$PATH"

export MKL_CBWR="AVX2"
export I_MPI_FABRICS=shm:ofi
ulimit -s unlimited

python3 dos_builder_fermi_align.py > log_dos_builder_fermi_align
