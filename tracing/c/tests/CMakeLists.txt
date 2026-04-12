cmake_minimum_required(VERSION 3.18)
project(tracing_tests VERSION 1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)

include(CheckCSourceCompiles)

find_package(Threads)

set(MPSC_TEST_SRC test_mpsc.c)
set(MPSC_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/..)

# --- Sanitizer detection ---
# check_c_compiler_flag does not pass the flag to the linker, so sanitizer
# checks fail on toolchains that need the runtime at link time (e.g. Apple Clang).
# Use check_c_source_compiles with both compile and link flags instead.

macro(check_sanitizer_support flag result)
    set(CMAKE_REQUIRED_FLAGS "${flag}")
    set(CMAKE_REQUIRED_LINK_OPTIONS "${flag}")
    check_c_source_compiles("int main(void){return 0;}" ${result})
    unset(CMAKE_REQUIRED_FLAGS)
    unset(CMAKE_REQUIRED_LINK_OPTIONS)
endmacro()

# --- Plain build ---
add_executable(test_mpsc ${MPSC_TEST_SRC})
target_include_directories(test_mpsc PRIVATE ${MPSC_INCLUDE_DIR})
if(Threads_FOUND)
    target_link_libraries(test_mpsc PRIVATE Threads::Threads)
endif()

if(MSVC)
    target_compile_options(test_mpsc PRIVATE /W4 /experimental:c11atomics)
else()
    target_compile_options(test_mpsc PRIVATE -Wall -Wextra -g)
endif()

# --- ASAN + UBSAN ---
if(MSVC)
    check_sanitizer_support("/fsanitize:address" HAS_ASAN)
    if(HAS_ASAN)
        add_executable(test_mpsc_asan ${MPSC_TEST_SRC})
        target_include_directories(test_mpsc_asan PRIVATE ${MPSC_INCLUDE_DIR})
        target_compile_options(test_mpsc_asan PRIVATE /fsanitize:address /W4 /experimental:c11atomics /Zi)
        target_link_options(test_mpsc_asan PRIVATE /fsanitize:address)
        if(Threads_FOUND)
            target_link_libraries(test_mpsc_asan PRIVATE Threads::Threads)
        endif()
    endif()
else()
    check_sanitizer_support("-fsanitize=address" HAS_ASAN)
    check_sanitizer_support("-fsanitize=undefined" HAS_UBSAN)
    check_sanitizer_support("-fsanitize=thread" HAS_TSAN)

    if(HAS_ASAN OR HAS_UBSAN)
        set(ASAN_FLAGS "")
        if(HAS_ASAN AND HAS_UBSAN)
            set(ASAN_FLAGS "-fsanitize=address,undefined")
        elseif(HAS_ASAN)
            set(ASAN_FLAGS "-fsanitize=address")
        else()
            set(ASAN_FLAGS "-fsanitize=undefined")
        endif()

        add_executable(test_mpsc_asan ${MPSC_TEST_SRC})
        target_include_directories(test_mpsc_asan PRIVATE ${MPSC_INCLUDE_DIR})
        target_compile_options(test_mpsc_asan PRIVATE ${ASAN_FLAGS} -fno-omit-frame-pointer -O1 -g -Wall -Wextra)
        target_link_options(test_mpsc_asan PRIVATE ${ASAN_FLAGS})
        if(Threads_FOUND)
            target_link_libraries(test_mpsc_asan PRIVATE Threads::Threads)
        endif()
    endif()

    # TSAN cannot be combined with ASAN
    if(HAS_TSAN)
        add_executable(test_mpsc_tsan ${MPSC_TEST_SRC})
        target_include_directories(test_mpsc_tsan PRIVATE ${MPSC_INCLUDE_DIR})
        target_compile_options(test_mpsc_tsan PRIVATE -fsanitize=thread -fno-omit-frame-pointer -O1 -g -Wall -Wextra)
        target_link_options(test_mpsc_tsan PRIVATE -fsanitize=thread)
        if(Threads_FOUND)
            target_link_libraries(test_mpsc_tsan PRIVATE Threads::Threads)
        endif()
    endif()
endif()

# --- CTest ---
enable_testing()
add_test(NAME mpsc COMMAND test_mpsc)
if(TARGET test_mpsc_asan)
    add_test(NAME mpsc_asan COMMAND test_mpsc_asan)
endif()
if(TARGET test_mpsc_tsan)
    add_test(NAME mpsc_tsan COMMAND test_mpsc_tsan)
endif()
