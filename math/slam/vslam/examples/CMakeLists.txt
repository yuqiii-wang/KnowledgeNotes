project(vslam_tutorial)
MESSAGE("Project source dir: " ${PROJECT_SOURCE_DIR}) 

cmake_minimum_required(VERSION 3.16)
set (CMAKE_CXX_STANDARD 20)

find_package( OpenCV REQUIRED )
MESSAGE("OpenCV version: " ${OpenCV_VERSION})
MESSAGE("OpenCV location: " ${OpenCV_INCLUDE_DIRS})
IF (${OpenCV_VERSION} LESS_EQUAL 4.5 OR ${OpenCV_VERSION} STREQUAL "")
    MESSAGE( FATAL_ERROR "Some old OpenCV versions have patent issues, please install the latest one" )
ENDIF()

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable( pnp src/pnp.cpp )
add_executable( bundle_adjustment src/bundle_adjustment.cpp )

target_link_libraries(pnp ${OpenCV_LIBRARIES} )
target_link_libraries(bundle_adjustment ${OpenCV_LIBRARIES} )