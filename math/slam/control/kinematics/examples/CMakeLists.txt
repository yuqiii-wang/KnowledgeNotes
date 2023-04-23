cmake_minimum_required(VERSION 3.23)

project(kinematics_tutorials)
MESSAGE("Project source dir: " ${PROJECT_SOURCE_DIR}) 

## Must use c++ 17, c++20 got operator== definition error in Eigen
set (CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE debug)


find_package( OpenCV REQUIRED )
find_package( g2o REQUIRED )
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${g2o_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/include/eigen3)
include_directories(${Ceres_INCLUDE_DIRS} ) 

add_executable( sim3 src/sim3.cpp )
target_link_libraries(sim3 ${OpenCV_LIBRARIES}
                        ${Eigen_LIBRARIES}
                        ${g2o_LIBRARIES}
                        ${CERES_LIBRARIES}
                        )
