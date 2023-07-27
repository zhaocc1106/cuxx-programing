cmake_minimum_required(VERSION 3.26)
project(dgemm LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(dgemm main.cpp)
target_include_directories(dgemm PRIVATE /usr/local/cuda/include)
target_link_directories(dgemm PRIVATE /usr/local/cuda/lib64)
target_link_libraries(dgemm PRIVATE cublas cudart)