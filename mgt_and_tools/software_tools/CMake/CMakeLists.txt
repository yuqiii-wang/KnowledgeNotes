cmake_minimum_required (VERSION 3.1.0)
project (sp_trader_run)
set (CMAKE_CXX_STANDARD 11)

# include_directories is for headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# link_directories is for 3rd party built binaries to be found
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

# locate 3rd party built binaries
find_library(APIWRAPPER_PATH apiwrapper HINTS ${CMAKE_CURRENT_SOURCE_DIR} PATH_SUFFIXES lib)

# Create a library called "ApiTester" which includes the source file "ApiTester.cxx".
# The extension is already found. Any number of sources could be listed here.
add_library (ApiTester SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/ApiTester.cpp)

# Add executable called "${PROJECT_NAME}" that is built from the source files
add_executable (${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

# Link the executable to all libraries. 
target_link_libraries (${PROJECT_NAME} ApiTester ${APIWRAPPER_PATH})