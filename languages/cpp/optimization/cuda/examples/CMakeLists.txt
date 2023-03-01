project(cuda_proj LANGUAGES CXX CUDA) # enable c++ and cuda
MESSAGE("Project source dir: " ${PROJECT_SOURCE_DIR}) 

# set your own g++, 
# cmake default compiler might not support std=c++20
set(CMAKE_C_COMPILER /usr/bin/g++)
set(CMAKE_CXX_COMPILER /usr/bin/g++)

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE debug)
set(CMAKE_CXX_FLAGS "-mavx2")

find_package(benchmark REQUIRED HINTS /home/yuqi/lib/cmake/)

add_executable(cuda_1d src/cuda_1d.cu)
# add_executable(cuda_1dAsync src/cuda_1dAsync.cu)
add_executable(cuda_3d1d src/cuda_3d1d.cu)
add_executable(cuda_3d1dPinnedMem src/cuda_3d1dPinnedMem.cu)
# add_executable(cuda_cublas_add src/cuda_cublas_add.cu)

target_link_libraries(cuda_1d benchmark::benchmark)