# cmake needs this line
cmake_minimum_required(VERSION 3.1)

PROJECT(imread)

find_package(OpenCV REQUIRED)

MESSAGE(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")


add_executable(imread imread.cpp)

target_link_libraries(imread ${OpenCV_LIBS})




