cmake_minimum_required(VERSION 3.0.2)
project(btc_desc)

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_BUILD_TYPE "Release")

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads)
find_package(Ceres REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cpp)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pybind)
