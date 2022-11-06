project(vslam_tutorial)
MESSAGE("Project source dir: " ${PROJECT_SOURCE_DIR}) 

cmake_minimum_required(VERSION 3.16)

## Must use c++ 17, c++20 got operator== definition error in Eigen
set (CMAKE_CXX_STANDARD 17)

find_package( OpenCV REQUIRED )
MESSAGE("OpenCV version: " ${OpenCV_VERSION})
MESSAGE("OpenCV location: " ${OpenCV_INCLUDE_DIRS})
IF (${OpenCV_VERSION} LESS_EQUAL 4.5 OR ${OpenCV_VERSION} STREQUAL "")
    MESSAGE( FATAL_ERROR "Some old OpenCV versions have patent issues, please install the latest one" )
ENDIF()

find_package(Eigen3 3.4 REQUIRED)
MESSAGE("Eigen3 version: " ${Eigen3_VERSION})

find_package(g2o 1.0 REQUIRED)
find_package(Sophus 22.4 REQUIRED)
find_package(fmt 9.1 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/include/eigen3)
include_directories(${g2o_INCLUDE_DIRS})
include_directories(${Sophus_INCLUDE_DIRS})
include_directories(${fmt_INCLUDE_DIRS} ) 


add_executable( icp src/icp.cpp )

target_link_libraries(icp ${OpenCV_LIBRARIES}
                        ${Eigen_LIBRARIES}
                        ${g2o_LIBRARIES}
                        ${Sophus_LIBRARIES} 
                        fmt)