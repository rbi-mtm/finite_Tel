#PBS -q cpu
#PBS -l select=64:mem=1500MB
##PBS -l place=free
#PBS -l place=pack
#PBS -j oe
#PBS -N aims

cd ${PBS_O_WORKDIR}

MPI_NUM_PROCESSES=$(cat ${PBS_NODEFILE} | wc -l)

module load scientific/abinit/9.8.4-gnu
module load libs/scalapack/2.2.0-gnu

EXE=~/vasp/vasp.6.4.2/bin/vasp_std

export OMP_NUM_THREADS=1
echo $EXE

mpiexec -np ${MPI_NUM_PROCESSES} $EXE > log.out 2>&1
