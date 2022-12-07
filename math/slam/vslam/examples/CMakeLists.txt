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
find_package(Ceres 2.2 REQUIRED)
find_package(glog 0.6 REQUIRED)


include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/include/eigen3)
include_directories(${g2o_INCLUDE_DIRS})
include_directories(${Sophus_INCLUDE_DIRS})
include_directories(${fmt_INCLUDE_DIRS} ) 
include_directories(${Ceres_INCLUDE_DIRS} ) 

add_executable( pnp src/pnp.cpp )
add_executable( ba_gauss_newton src/ba_gauss_newton.cpp )
add_executable( ba_g2o src/ba_g2o.cpp )
add_executable( ba_ceres src/ba_ceres.cpp )
add_executable( optical_flow src/optical_flow.cpp )

target_link_libraries(pnp ${OpenCV_LIBRARIES} )

target_link_libraries(ba_gauss_newton ${OpenCV_LIBRARIES}
                        ${Eigen_LIBRARIES}
                        ${g2o_LIBRARIES}
                        ${Sophus_LIBRARIES} 
                        fmt)

target_link_libraries(ba_g2o ${OpenCV_LIBRARIES}
                        ${Eigen_LIBRARIES}
                        ${g2o_LIBRARIES}
                        ${Sophus_LIBRARIES} 
                        fmt
                        ${CERES_LIBRARIES}
                        glog::glog)

target_link_libraries(ba_ceres ${OpenCV_LIBRARIES}
                        ${Eigen_LIBRARIES}
                        ${g2o_LIBRARIES}
                        ${Sophus_LIBRARIES} 
                        fmt
                        ${CERES_LIBRARIES}
                        glog::glog)

target_link_libraries(optical_flow ${OpenCV_LIBRARIES}
                        ${Eigen_LIBRARIES}
                        ${g2o_LIBRARIES}
                        ${Sophus_LIBRARIES} 
                        fmt)

FILE(GLOB images ${PROJECT_SOURCE_DIR}/resources/traffic*.png)
FILE(COPY ${images}
        DESTINATION ${PROJECT_SOURCE_DIR}/build/ )
FILE(COPY resources/landmarks.txt
        DESTINATION ${PROJECT_SOURCE_DIR}/build/ )