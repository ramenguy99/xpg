cmake_minimum_required(VERSION 3.18)

project(example_cpp VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

find_package(Threads)
find_package(SQLite3 QUIET)

add_executable(example_cpp main.cpp worker.cpp tracer_impl.cpp)
target_include_directories(example_cpp PRIVATE ${CMAKE_SOURCE_DIR}/..)

if(Threads_FOUND)
    target_link_libraries(example_cpp PRIVATE Threads::Threads)
endif()

if(SQLite3_FOUND)
    target_compile_definitions(example_cpp PRIVATE TRACER_SQLITE_ENABLED)
    target_link_libraries(example_cpp PRIVATE SQLite::SQLite3)
endif()
