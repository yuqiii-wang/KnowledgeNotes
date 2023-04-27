cmake_minimum_required(VERSION 3.16)

project(vslam_tutorial)
MESSAGE("Project source dir: " ${PROJECT_SOURCE_DIR}) 


## Must use c++ 17, c++20 got operator== definition error in Eigen
set (CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE debug)

find_package( OpenCV REQUIRED )

find_package(Eigen3 3.3 REQUIRED)

find_package(g2o 1.0 REQUIRED)
find_package(Sophus REQUIRED)
find_package(fmt REQUIRED)
find_package(Ceres REQUIRED)
# find_package(glog REQUIRED)


include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/include/eigen3)
include_directories(${g2o_INCLUDE_DIRS})
include_directories(${Sophus_INCLUDE_DIRS})
include_directories(${fmt_INCLUDE_DIRS} ) 
include_directories(${Ceres_INCLUDE_DIRS} ) 

add_executable( pnp src/pnp.cpp )
add_executable( ba_gauss_newton src/ba_gauss_newton.cpp )
# add_executable( ba_g2o src/ba_g2o.cpp )
add_executable( ba_ceres src/ba_ceres.cpp )
add_executable( optical_flow src/optical_flow.cpp )
add_executable( triangulation_depth_estimate src/triangulation_depth_estimate.cpp )

target_link_libraries(pnp ${OpenCV_LIBRARIES} )

target_link_libraries(ba_gauss_newton ${OpenCV_LIBRARIES}
                        ${Eigen_LIBRARIES}
                        ${g2o_LIBRARIES}
                        ${Sophus_LIBRARIES} 
                        fmt)

# target_link_libraries(ba_g2o ${OpenCV_LIBRARIES}
#                         ${Eigen_LIBRARIES}
#                         ${g2o_LIBRARIES}
#                         ${Sophus_LIBRARIES} 
#                         fmt
#                         ${CERES_LIBRARIES}
#                         )

target_link_libraries(ba_ceres ${OpenCV_LIBRARIES}
                        ${Eigen_LIBRARIES}
                        ${g2o_LIBRARIES}
                        ${Sophus_LIBRARIES} 
                        fmt
                        ${CERES_LIBRARIES}
                        )

target_link_libraries(optical_flow ${OpenCV_LIBRARIES}
                        ${Eigen_LIBRARIES}
                        ${g2o_LIBRARIES}
                        ${Sophus_LIBRARIES} 
                        fmt)

target_link_libraries(triangulation_depth_estimate ${OpenCV_LIBRARIES}
                        ${Eigen_LIBRARIES}
                        ${g2o_LIBRARIES}
                        ${Sophus_LIBRARIES} 
                        fmt)

FILE(GLOB images ${PROJECT_SOURCE_DIR}/resources/*.png)
FILE(COPY ${images}
        DESTINATION ${PROJECT_SOURCE_DIR}/build/ )
FILE(COPY resources/landmarks.txt
        DESTINATION ${PROJECT_SOURCE_DIR}/build/ )