cmake_minimum_required(VERSION 3.1)

PROJECT(vid-arg)

find_package(OpenCV REQUIRED)

add_executable(vid-arg main.cpp)

target_link_libraries(vid-arg ${OpenCV_LIBS})

