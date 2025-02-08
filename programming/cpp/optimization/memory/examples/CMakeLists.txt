project(googletest_tutorial)
MESSAGE("Project source dir: " ${PROJECT_SOURCE_DIR}) 

cmake_minimum_required(VERSION 3.16)
set (CMAKE_CXX_STANDARD 20)

find_package(Threads REQUIRED)
set(CMAKE_CXX_FLAGS -pthread)

add_executable(alloc src/alloc.cpp)
add_executable(atomic_load_store src/atomic_load_store.cpp)
add_executable(mem_ord_relaxed src/mem_ord_relaxed.cpp)
add_executable(mem_ord_seq_cst src/mem_ord_seq_cst.cpp)