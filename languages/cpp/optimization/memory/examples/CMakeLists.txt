project(googletest_tutorial)
MESSAGE("Project source dir: " ${PROJECT_SOURCE_DIR}) 

cmake_minimum_required(VERSION 3.16)
set (CMAKE_CXX_STANDARD 20)

add_executable(main src/main.cpp)