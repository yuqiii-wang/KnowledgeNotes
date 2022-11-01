project(cv_tutorials)
MESSAGE("Project source dir: " ${PROJECT_SOURCE_DIR}) 

cmake_minimum_required(VERSION 3.16)
set (CMAKE_CXX_STANDARD 20)

find_package( OpenCV REQUIRED )
MESSAGE("OpenCV version: " ${OpenCV_VERSION})
MESSAGE("OpenCV location: " ${OpenCV_INCLUDE_DIRS})
IF (${OpenCV_VERSION} LESS_EQUAL 4.5 OR ${OpenCV_VERSION} STREQUAL "")
    MESSAGE( FATAL_ERROR "Some old OpenCV versions have patent issues, please install the latest one" )
ENDIF()

find_package(PCL REQUIRED)
find_package(Eigen REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${Eigen_INCLUDE_DIRS})

add_executable( Triangulation src/Triangulation.cpp )

target_link_libraries(Triangulation ${OpenCV_LIBRARIES} 
                                    ${Eigen_LIBRARIES}
                                    ${PCL_LIBRARIES})

FILE(COPY ${PROJECT_SOURCE_DIR}/resources/street_view_left.png 
        ${PROJECT_SOURCE_DIR}/resources/street_view_right.png 
        DESTINATION ${PROJECT_SOURCE_DIR}/build/ )