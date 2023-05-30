project(googletest_tutorial)
MESSAGE("Project source dir: " ${PROJECT_SOURCE_DIR}) 

# set your own g++, 
# cmake default compiler might not support std=c++20
set(CMAKE_C_COMPILER /usr/bin/g++)
set(CMAKE_CXX_COMPILER /usr/bin/g++)

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE debug)

set(CMAKE_CXX_FLAGS "-fsanitize=address")


add_executable(lockfree_mq_list src/lockfree_mq_list.cpp)
add_executable(lockfree_mq_array src/lockfree_mq_array.cpp)
