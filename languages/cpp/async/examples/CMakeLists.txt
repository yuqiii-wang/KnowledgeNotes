project(googletest_tutorial)
MESSAGE("Project source dir: " ${PROJECT_SOURCE_DIR}) 

# set your own g++, 
# cmake default compiler might not support std=c++20
set(CMAKE_C_COMPILER /usr/bin/g++)
set(CMAKE_CXX_COMPILER /usr/bin/g++)

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE debug)
set(CMAKE_CXX_FLAGS "-pthread -fcoroutines")

add_executable(deadlock_debug src/deadlock_debug.cpp)
add_executable(future_and_promise src/future_and_promise.cpp)
add_executable(shared_lock src/shared_lock.cpp)
add_executable(coroutine_co_await src/coroutine_co_await.cpp)
add_executable(coroutine_co_yield src/coroutine_co_yield.cpp)
add_executable(coroutine_co_return src/coroutine_co_return.cpp)
add_executable(coroutine_fibonacci src/coroutine_fibonacci.cpp)
