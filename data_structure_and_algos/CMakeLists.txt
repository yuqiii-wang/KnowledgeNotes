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


add_executable(binary_tree_run data_structs/tree/binary_tree_run.cpp)

add_executable(findMedianFromTwoSortedVectors algos/binary_search/findMedianFromTwoSortedVectors.cpp)

add_executable(zigzagConversion data_structs/vector/zigzagConversion.cpp)
add_executable(deleteOddElemsFromAVector data_structs/vector/deleteOddElemsFromAVector.cpp)

add_executable(reverseInt data_structs/list/reverseInt.cpp)

add_executable(bfs_and_dfs algos/search/bfs_and_dfs.cpp)

add_executable(bubble_sort algos/sort/bubble_sort.cpp)
add_executable(quick_sort algos/sort/quick_sort.cpp)