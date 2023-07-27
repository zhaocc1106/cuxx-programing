cmake_minimum_required(VERSION 3.26)
project(1_based_index LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(1_based_index main.cpp)
target_include_directories(1_based_index PRIVATE /usr/local/cuda/include)
target_link_directories(1_based_index PRIVATE /usr/local/cuda/lib64)
target_link_libraries(1_based_index PRIVATE cublas cudart)