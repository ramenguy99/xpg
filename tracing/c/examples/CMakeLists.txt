cmake_minimum_required(VERSION 3.18)

project(tracing_example VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

if(MSVC)
    add_compile_options(/experimental:c11atomics)
endif()

add_executable(client_c
    example_client.c
)
target_include_directories(client_c PRIVATE ${CMAKE_SOURCE_DIR}/..)

add_executable(server_c
    example_server.c
)
target_include_directories(server_c PRIVATE ${CMAKE_SOURCE_DIR}/..)

# --- Tracer examples ---

find_package(SQLite3 QUIET)

# C tracer example
add_executable(example_tracer_c example_tracer.c)
target_include_directories(example_tracer_c PRIVATE ${CMAKE_SOURCE_DIR}/..)
if(SQLite3_FOUND)
    target_compile_definitions(example_tracer_c PRIVATE TRACER_SQLITE_ENABLED)
    target_link_libraries(example_tracer_c PRIVATE SQLite::SQLite3)
endif()

# C++ tracer example
add_executable(example_tracer_cpp example_tracer.cpp)
target_include_directories(example_tracer_cpp PRIVATE ${CMAKE_SOURCE_DIR}/..)
if(SQLite3_FOUND)
    target_compile_definitions(example_tracer_cpp PRIVATE TRACER_SQLITE_ENABLED)
    target_link_libraries(example_tracer_cpp PRIVATE SQLite::SQLite3)
endif()