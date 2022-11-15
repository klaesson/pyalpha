#!/bin/bash -l
			
#SBATCH -p cops	    # Partition
#SBATCH -N 2     	# Number of nodes
#SBATCH -J t_meep	# Job name
#SBATCH -t 6-00:00:00	# Wall time (days-H:M:S)
#SBATCH -o slurm-%j.out	# Job output file

conda activate pmp
mkdir out
ns=128
name=overlap_effective_medium

mpirun -n $ns python3 $name.py
find . -type f -name "*.h5" -delete
