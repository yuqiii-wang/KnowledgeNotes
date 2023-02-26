project(googletest_tutorial)
MESSAGE("Project source dir: " ${PROJECT_SOURCE_DIR}) 

# set your own g++, 
# cmake default compiler might not support std=c++20
set(CMAKE_C_COMPILER /usr/bin/g++)
set(CMAKE_CXX_COMPILER /usr/bin/g++)

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE debug)

set(COMPILE_FLAGS "-fsanitize=address")

add_executable(dijkstra src/dijkstra.cpp)