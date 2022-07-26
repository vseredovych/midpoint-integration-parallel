cmake_minimum_required(VERSION 3.22)
project(untitled)


# openmpi
find_package(MPI)
include_directories(${MPI_INCLUDE_PATH})

SET(CMAKE_CXX_COMPILER mpicxx)
#SET(CMAKE_C_COMPILER  mpicc)
# openmpi

set(CMAKE_CXX_STANDARD 20)

add_executable(untitled main.cpp)
