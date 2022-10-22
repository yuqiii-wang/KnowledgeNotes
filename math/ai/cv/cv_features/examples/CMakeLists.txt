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

find_package(Boost REQUIRED)
MESSAGE("Boost version: " ${Boost_VERSION})
MESSAGE("Boost location: " ${Boost_INCLUDE_DIRS})
IF (${Boost_VERSION} LESS_EQUAL 1.75 OR ${Boost_VERSION} STREQUAL "")
    MESSAGE( FATAL_ERROR "Old Boost version Error, please install the latest one" )
ENDIF()

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

add_executable( CannyDetector src/CannyDetector.cpp )
add_executable( SIFT src/SIFT.cpp )
add_executable( ORB src/ORB.cpp )
add_executable( HaarWavelet src/HaarWavelet.cpp )

target_link_libraries(CannyDetector ${OpenCV_LIBRARIES} )
target_link_libraries(SIFT ${OpenCV_LIBRARIES} )
target_link_libraries(ORB ${OpenCV_LIBRARIES} )
target_link_libraries(HaarWavelet ${Boost_LIBRARIES} )

FILE(COPY ${PROJECT_SOURCE_DIR}/resource/lenna.png 
    DESTINATION ${PROJECT_SOURCE_DIR}/build/ )