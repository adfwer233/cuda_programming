cmake_minimum_required(VERSION 3.8)

project(CUDA_Program VERSION 1.0 DESCRIPTION "project for learning cuda" LANGUAGES CUDA CXX)

add_executable(gemm gemm.cu)