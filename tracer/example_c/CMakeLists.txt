cmake_minimum_required(VERSION 3.18)

project(example_c VERSION 1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)

find_package(Threads)
find_package(SQLite3 QUIET)

add_executable(example_c main.c worker.c tracer_impl.c)
target_include_directories(example_c PRIVATE ${CMAKE_SOURCE_DIR}/..)

if(MSVC)
    target_compile_options(example_c PRIVATE /experimental:c11atomics)
endif()

if(Threads_FOUND)
    target_link_libraries(example_c PRIVATE Threads::Threads)
endif()

if(SQLite3_FOUND)
    target_compile_definitions(example_c PRIVATE TRACER_SQLITE_ENABLED)
    target_link_libraries(example_c PRIVATE SQLite::SQLite3)
endif()
