project(googletest_tutorial)
MESSAGE("Project source dir: " ${PROJECT_SOURCE_DIR}) 

cmake_minimum_required(VERSION 3.16)
set (CMAKE_CXX_STANDARD 20)

set(THREADPOOL_AUTO_GROW 1)

find_package(Threads REQUIRED)

add_executable(mem_leak_test_simple src/mem_leak_test_simple.cpp)
add_executable(mem_leak_test src/mem_leak_test.cpp)

target_compile_options(mem_leak_test_simple PRIVATE -Wall)
target_compile_options(mem_leak_test_simple PRIVATE -ggdb3)