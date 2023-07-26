cmake_minimum_required(VERSION 3.26)
project(cublaslt LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(cublaslt main.cpp)
target_include_directories(cublaslt PRIVATE /usr/local/cuda/include)
target_link_directories(cublaslt PRIVATE /usr/local/cuda/lib64)
target_link_libraries(cublaslt PRIVATE cublas cudart cublasLt)