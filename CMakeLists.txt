cmake_minimum_required(VERSION 3.8)

project(CUDA_Program VERSION 1.0 DESCRIPTION "project for learning cuda" LANGUAGES CUDA CXX)

add_executable(gemm gemm.cu)

add_executable(gemm_wmma gemm_wmma.cu)


target_compile_options(gemm_wmma PRIVATE "-arch=compute_86")