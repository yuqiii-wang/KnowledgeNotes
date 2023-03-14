project(googletest_tutorial)
MESSAGE("Project source dir: " ${PROJECT_SOURCE_DIR}) 

# set your own g++, 
# cmake default compiler might not support std=c++20
set(CMAKE_C_COMPILER /usr/bin/g++)
set(CMAKE_CXX_COMPILER /usr/bin/g++)

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE debug)

find_package(Threads REQUIRED)
set(CMAKE_CXX_FLAGS "-mavx2 -pthread")

find_package(benchmark REQUIRED HINTS /home/yuqi/lib/cmake/)

add_executable(example_simd src/example_simd.cpp)
add_executable(branching src/branching.cpp)
add_executable(return_val_opt src/return_val_opt.cpp)
add_executable(thread_aff src/thread_aff.cpp)
add_executable(read_tsc src/read_tsc.cpp)
add_executable(pointer_vs_reference src/pointer_vs_reference.cpp)

target_link_libraries(read_tsc benchmark::benchmark)
target_link_libraries(branching benchmark::benchmark)
target_link_libraries(example_simd benchmark::benchmark)
target_link_libraries(pointer_vs_reference benchmark::benchmark)