#!/bin/bash

set -e

nproc=$1

# Compile the C file
mpicc main.cpp -o test_mpi

# Run compiled test_mpi.c file
mpirun -np $nproc ./test_mpi
