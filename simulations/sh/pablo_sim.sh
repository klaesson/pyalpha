#!/bin/bash -l

mkdir out

mpirun -n 72 python3 overlap_effective_medium.py

find . -type f -name "*.h5" -delete