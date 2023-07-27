cmake_minimum_required(VERSION 3.26)
project(cusparse_demo LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(cusparse_demo main.cpp)
target_include_directories(cusparse_demo PRIVATE /usr/local/cuda/include)
target_link_directories(cusparse_demo PRIVATE /usr/local/cuda/lib64)
target_link_libraries(cusparse_demo PRIVATE cublas cudart cublasLt cusparse)