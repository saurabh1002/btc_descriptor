cmake_minimum_required(VERSION 3.16...3.26)
project(btcdesc_cpp)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads)
find_package(Ceres REQUIRED)

include(cmake/CompilerOptions.cmake)

add_subdirectory(btcdesc)

