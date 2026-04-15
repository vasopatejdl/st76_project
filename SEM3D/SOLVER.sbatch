#!/bin/bash
#SBATCH --job-name=SOLVER 
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --time=00:30:00
#SBATCH --output=ro.txt 
#SBATCH --error=re.txt 
#SBATCH --partition=cpu_prod
#SBATCH --qos=16nodespu

module load sem3d/2024.02.22/intel-2021.9.0-intel-oneapi-mpi

mpirun -n $SLURM_NTASKS sem3d.exe > output.solver
