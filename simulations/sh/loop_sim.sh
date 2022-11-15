#!/bin/bash -l
			
#SBATCH -p cops    # Partition
#SBATCH -N 3		# Number of nodes
#SBATCH -J t_meep	# Job name
#SBATCH -t 4-00:00:00	# Wall time (days-H:M:S)
#SBATCH -o slurm-%j.out	# Job output file

conda activate pmp

name="lattice_2D_B_tuning_wires"
ns=192

var="x"
j=0
mkdir $var
cd $var
mkdir out
mpirun -n $ns python3 ../$name.py $var $j
find . -type f -name "*.h5" -delete
cd ..


var="y"
mkdir $var
cd $var
for i in 0 2 4 6 8; do
    mkdir setup_ny_$i
    cd setup_ny_$i
    mkdir out
    mpirun -n $ns python3 ../../$name.py $var $i
    find . -type f -name "*.h5" -delete
    cd ..    
done;

