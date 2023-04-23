project(SophusExample)

cmake_minimum_required(VERSION 3.16)

# should first install eigen, fmt and sophus
find_package( Sophus REQUIRED )
find_package( fmt REQUIRED )

include_directories( ${fmt_INCLUDE_DIRS} ) 
include_directories( ${Sophus_INCLUDE_DIRS} ) 
include_directories(/usr/include/eigen3)

add_executable( SophusExample src/SophusExample.cpp )

target_link_libraries(SophusExample 
                                ${Sophus_LIBRARIES} 
                                fmt)