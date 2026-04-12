cmake_minimum_required(VERSION 3.18)

project(tracing_app VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

add_executable(app main.cpp worker.cpp)
target_include_directories(app PRIVATE ${CMAKE_SOURCE_DIR}/..)
