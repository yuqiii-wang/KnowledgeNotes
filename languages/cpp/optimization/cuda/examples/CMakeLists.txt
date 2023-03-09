project(cuda_proj LANGUAGES CXX CUDA) # enable c++ and cuda
MESSAGE("Project source dir: " ${PROJECT_SOURCE_DIR}) 

# set your own cuda
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE debug)
set(CMAKE_CXX_FLAGS "-mavx2 -lcublas")


find_package(benchmark REQUIRED HINTS /home/yuqi/lib/cmake/)
find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )


add_executable(cuda_1d src/cuda_1d.cu)
# add_executable(cuda_1dAsync src/cuda_1dAsync.cu)
add_executable(cuda_3d1d src/cuda_3d1d.cu)
add_executable(cuda_3d1dPinnedMem src/cuda_3d1dPinnedMem.cu)
# add_executable(cuda_cublas_add src/cuda_cublas_add.cu)

add_executable(ray_tracing src/ray_tracing.cu)
add_executable(ray_tracing_withConstant src/ray_tracing_withConstant.cu)

add_executable(heat_transfer src/heat_transfer.cu)
add_executable(heat_transfer_withTexture src/heat_transfer_withTexture.cu)

target_link_libraries( heat_transfer ${OpenCV_LIBS} )
target_link_libraries( heat_transfer_withTexture ${OpenCV_LIBS} )

target_link_libraries( ray_tracing ${OpenCV_LIBS} )
target_link_libraries( ray_tracing_withConstant ${OpenCV_LIBS} )
target_link_libraries(cuda_1d benchmark::benchmark)
