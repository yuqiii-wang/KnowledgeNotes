project(googletest_tutorial)
MESSAGE("Project source dir: " ${PROJECT_SOURCE_DIR}) 

# set your own g++, 
# cmake default compiler might not support std=c++20
set(CMAKE_C_COMPILER /usr/bin/g++)
set(CMAKE_CXX_COMPILER /usr/bin/g++)

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE debug)

set(CMAKE_CXX_FLAGS "-fsanitize=address -fpermissive")


add_executable(crtp src/crtp.cpp)
add_executable(crtp_covariant src/crtp_covariant.cpp)
add_executable(sfinae_concept_test src/sfinae_concept_test.cpp)
add_executable(template_get_private src/template_get_private.cpp)
