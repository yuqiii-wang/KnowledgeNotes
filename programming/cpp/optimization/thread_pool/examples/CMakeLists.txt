project(googletest_tutorial)
MESSAGE("Project source dir: " ${PROJECT_SOURCE_DIR}) 

cmake_minimum_required(VERSION 3.16)
set (CMAKE_CXX_STANDARD 20)

set(THREADPOOL_AUTO_GROW 1)

find_package(Threads REQUIRED)

add_executable(main src/main.cpp)

target_link_libraries(main PUBLIC Threads::Threads)