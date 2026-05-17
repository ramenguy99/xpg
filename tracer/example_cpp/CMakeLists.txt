cmake_minimum_required(VERSION 3.18)

project(example_cpp VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

add_executable(example_cpp main.cpp worker.cpp tracer_impl.cpp)
target_include_directories(example_cpp PRIVATE ${CMAKE_SOURCE_DIR}/..)
