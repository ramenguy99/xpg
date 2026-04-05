cmake_minimum_required(VERSION 3.18)

project(tracing_example VERSION 1.0 LANGUAGES C CXX)

add_executable(client_c
    example_client.c
)
target_include_directories(client_c PRIVATE ${CMAKE_SOURCE_DIR}/..)

add_executable(server_c
    example_server.c
)
target_include_directories(server_c PRIVATE ${CMAKE_SOURCE_DIR}/..)
