cmake_minimum_required(VERSION 3.22)
project(opencv LANGUAGES CXX)

find_package(OpenCV REQUIRED)
                    
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(opencv OpenCV.cpp)  
target_link_libraries(opencv ${OpenCV_LIBS})       