cmake_minimum_required(VERSION 3.18)
project(tracing_tests VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

include(CheckCSourceCompiles)

find_package(Threads)

set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/..)

# --- Sanitizer detection ---
macro(check_sanitizer_support flag result)
    set(CMAKE_REQUIRED_FLAGS "${flag}")
    set(CMAKE_REQUIRED_LINK_OPTIONS "${flag}")
    check_c_source_compiles("int main(void){return 0;}" ${result})
    unset(CMAKE_REQUIRED_FLAGS)
    unset(CMAKE_REQUIRED_LINK_OPTIONS)
endmacro()

if(NOT MSVC)
    check_sanitizer_support("-fsanitize=address" HAS_ASAN)
    check_sanitizer_support("-fsanitize=undefined" HAS_UBSAN)
    check_sanitizer_support("-fsanitize=thread" HAS_TSAN)

    if(HAS_ASAN AND HAS_UBSAN)
        set(ASAN_FLAGS "-fsanitize=address,undefined")
    elseif(HAS_ASAN)
        set(ASAN_FLAGS "-fsanitize=address")
    elseif(HAS_UBSAN)
        set(ASAN_FLAGS "-fsanitize=undefined")
    endif()
endif()

# --- SQLite detection ---
find_package(SQLite3 QUIET)

# --- Helper: add a test target with optional sanitizer variants ---
function(add_test_target name source lang)
    add_executable(${name} ${source})
    target_include_directories(${name} PRIVATE ${INCLUDE_DIR} ${CMAKE_SOURCE_DIR})
    if(Threads_FOUND)
        target_link_libraries(${name} PRIVATE Threads::Threads)
    endif()

    if(MSVC)
        if("${lang}" STREQUAL "CXX")
            target_compile_options(${name} PRIVATE /W4)
        else()
            target_compile_options(${name} PRIVATE /W4 /experimental:c11atomics)
        endif()
    else()
        target_compile_options(${name} PRIVATE -Wall -Wextra -g)
    endif()

    # ASAN variant
    if(ASAN_FLAGS)
        set(asan_name "${name}_asan")
        add_executable(${asan_name} ${source})
        target_include_directories(${asan_name} PRIVATE ${INCLUDE_DIR} ${CMAKE_SOURCE_DIR})
        target_compile_options(${asan_name} PRIVATE ${ASAN_FLAGS} -fno-omit-frame-pointer -O1 -g -Wall -Wextra)
        target_link_options(${asan_name} PRIVATE ${ASAN_FLAGS})
        if(Threads_FOUND)
            target_link_libraries(${asan_name} PRIVATE Threads::Threads)
        endif()
    endif()

    # TSAN variant
    if(HAS_TSAN)
        set(tsan_name "${name}_tsan")
        add_executable(${tsan_name} ${source})
        target_include_directories(${tsan_name} PRIVATE ${INCLUDE_DIR} ${CMAKE_SOURCE_DIR})
        target_compile_options(${tsan_name} PRIVATE -fsanitize=thread -fno-omit-frame-pointer -O1 -g -Wall -Wextra)
        target_link_options(${tsan_name} PRIVATE -fsanitize=thread)
        if(Threads_FOUND)
            target_link_libraries(${tsan_name} PRIVATE Threads::Threads)
        endif()
    endif()
endfunction()

# --- Test targets ---
add_test_target(test_mpsc test_mpsc.c C)
add_test_target(test_socket test_socket.cpp CXX)
add_test_target(test_locks test_locks.cpp CXX)
add_test_target(test_ring test_ring.cpp CXX)
add_test_target(test_tracer test_tracer.cpp CXX)

# SQLite support for test_tracer
if(SQLite3_FOUND)
    target_compile_definitions(test_tracer PRIVATE TRACER_SQLITE_ENABLED)
    target_link_libraries(test_tracer PRIVATE SQLite::SQLite3)

    if(TARGET test_tracer_asan)
        target_compile_definitions(test_tracer_asan PRIVATE TRACER_SQLITE_ENABLED)
        target_link_libraries(test_tracer_asan PRIVATE SQLite::SQLite3)
    endif()
    if(TARGET test_tracer_tsan)
        target_compile_definitions(test_tracer_tsan PRIVATE TRACER_SQLITE_ENABLED)
        target_link_libraries(test_tracer_tsan PRIVATE SQLite::SQLite3)
    endif()
endif()

# --- CTest ---
enable_testing()

add_test(NAME mpsc COMMAND test_mpsc)
add_test(NAME socket COMMAND test_socket)
add_test(NAME locks COMMAND test_locks)
add_test(NAME ring COMMAND test_ring)
add_test(NAME tracer COMMAND test_tracer)

if(TARGET test_mpsc_asan)
    add_test(NAME mpsc_asan COMMAND test_mpsc_asan)
endif()
if(TARGET test_socket_asan)
    add_test(NAME socket_asan COMMAND test_socket_asan)
endif()
if(TARGET test_locks_asan)
    add_test(NAME locks_asan COMMAND test_locks_asan)
endif()
if(TARGET test_ring_asan)
    add_test(NAME ring_asan COMMAND test_ring_asan)
endif()
if(TARGET test_tracer_asan)
    add_test(NAME tracer_asan COMMAND test_tracer_asan)
endif()

if(TARGET test_mpsc_tsan)
    add_test(NAME mpsc_tsan COMMAND test_mpsc_tsan)
endif()
if(TARGET test_locks_tsan)
    add_test(NAME locks_tsan COMMAND test_locks_tsan)
endif()
if(TARGET test_tracer_tsan)
    add_test(NAME tracer_tsan COMMAND test_tracer_tsan)
endif()
