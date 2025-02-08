project(googletest_tutorial)
MESSAGE("Project source dir: " ${PROJECT_SOURCE_DIR}) 

cmake_minimum_required(VERSION 3.16)
set (CMAKE_CXX_STANDARD 20)

find_package( GTest REQUIRED )
find_package( benchmark REQUIRED )

find_library(LIBRT rt)
add_subdirectory(benchmark)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library (functions SHARED ${PROJECT_SOURCE_DIR}/src/test_functions.cpp)

add_executable(googletest ${PROJECT_SOURCE_DIR}/src/googletest.cpp)
add_executable(googletest_mem_leak ${PROJECT_SOURCE_DIR}/src/googletest_mem_leak.cpp)
add_executable(googlebenchmark ${PROJECT_SOURCE_DIR}/src/googlebenchmark.cpp
                                ${LIBRT})

target_link_libraries(googletest PUBLIC 
                        GTest::GTest
                        GTest::Main
                        functions)
target_link_libraries(googletest_mem_leak PUBLIC 
                        GTest::GTest
                        GTest::Main
                        )
target_link_libraries(googlebenchmark PUBLIC 
                        GTest::GTest
                        GTest::Main
                        benchmark::benchmark
                        )
                        
enable_testing()
add_test(NAME googletest_test COMMAND googletest)                 
add_test(NAME googletest_mem_leak_test COMMAND googletest_mem_leak --check_for_leaks)