cmake_minimum_required(VERSION 3.1)

PROJECT(gui-filter)

find_package(OpenCV REQUIRED)

add_executable(gui-filter main.cpp)

target_link_libraries(gui-filter ${OpenCV_LIBS})

