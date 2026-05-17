cmake_minimum_required(VERSION 3.18)

project(example_c VERSION 1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)

add_executable(example_c main.c worker.c tracer_impl.c)
target_include_directories(example_c PRIVATE ${CMAKE_SOURCE_DIR}/..)
