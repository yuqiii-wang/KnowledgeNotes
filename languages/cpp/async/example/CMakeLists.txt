project(googletest_tutorial)
MESSAGE("Project source dir: " ${PROJECT_SOURCE_DIR}) 

cmake_minimum_required(VERSION 3.16)
set (CMAKE_CXX_STANDARD 20)

set(THREADPOOL_AUTO_GROW 1)

find_package(Threads REQUIRED)

add_executable(deadlock_debug src/deadlock_debug.cpp)
add_executable(future_and_promise src/future_and_promise.cpp)
add_executable(shared_lock src/shared_lock.cpp)

target_link_libraries(deadlock_debug PUBLIC Threads::Threads)
target_link_libraries(shared_lock PUBLIC Threads::Threads)
target_link_libraries(future_and_promise PUBLIC Threads::Threads)