cmake_minimum_required(VERSION 3.5)

project(XGP VERSION 1.0 LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 20)

# XPG options
set(XPG_BUILD_APPS ON CACHE BOOL "Build XPG applications")
set(XPG_PYTHON ON CACHE BOOL "Build XPG python module")
set(XPG_MSVC_ANALYZE OFF CACHE BOOL "Build with /analyze when using MSVC (heavily increases build times)")

# Configure GLFW
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Build GLFW examples")
set(GLFW_BUILD_TESTS OFF CACHE BOOL "Build GLFW tests")

# Configure ZSTD
set(ZSTD_BUILD_STATIC ON)
set(ZSTD_BUILD_SHARED OFF)
set(ZSTD_BUILD_PROGRAMS OFF)
set(ZSTD_MULTITHREAD_SUPPORT OFF)
set(ZSTD_LEGACY_SUPPORT OFF)

# # Configure slang
# set(SLANG_ENABLE_CUDA OFF)
# set(SLANG_ENABLE_OPTIX OFF)
# set(SLANG_ENABLE_NVAPI OFF)
# set(SLANG_ENABLE_XLIB OFF)
# set(SLANG_ENABLE_SLANG_RHI OFF)
# # set(SLANG_ENABLE_TESTS OFF) broken?
# set(SLANG_ENABLE_EXAMPLES OFF)
# set(SLANG_ENABLE_REPLAYER OFF)
# set(SLANG_EMBED_STDLIB_SOURCE OFF)
# set(SLANG_EMBED_STDLIB ON)
# set(SLANG_LIB_TYPE STATIC)
# set(SLANG_SLANG_LLVM_FLAVOR DISABLE)
# add_subdirectory(ext/slang)

# Add external dependencies
add_subdirectory(ext/glfw)
add_subdirectory(ext/volk)
add_subdirectory(ext/glm)
add_subdirectory(ext/VulkanMemoryAllocator)
add_subdirectory(ext/zstd/build/cmake)

# Compiler specific
if (MSVC AND ${XPG_MSVC_ANALYZE})
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /analyze")
endif()

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-volatile -Wno-nullability-completeness -fPIC")
endif()

# Library
add_library(xpg STATIC
    src/include/xpg/array.h
    src/include/xpg/buffered_stream.h
    src/include/xpg/defines.h
    src/include/xpg/gfx.h
    src/include/xpg/graph.h
    src/include/xpg/gui.h
    src/include/xpg/hashmap.h
    src/include/xpg/log.h
    src/include/xpg/platform.h
    src/include/xpg/pool.h
    src/include/xpg/threading.h
    src/lib/gfx.cpp
    src/lib/platform.cpp
    src/lib/gui.cpp
)

target_include_directories(xpg PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>/src/include
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>/ext/imgui
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>/ext/atomic_queue/include
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(xpg PUBLIC volk_headers glfw glm VulkanMemoryAllocator)

if (MSVC OR CMAKE_C_SIMULATE_ID STREQUAL "MSVC")
    target_compile_definitions(xpg PUBLIC _CRT_SECURE_NO_WARNINGS)
endif()


# Add apps
if (XPG_BUILD_APPS)
    add_subdirectory(apps)
endif()


# Python module
if(XPG_PYTHON)

# Find python
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()
find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

# Add nanobind
add_subdirectory(ext/nanobind)

# Create python module
nanobind_add_module(pyxpg src/python/module.cpp)
target_link_libraries(pyxpg PRIVATE xpg)
# set_target_properties(pyxpg PROPERTIES OUTPUT_NAME "xpg")
nanobind_add_stub(pyxpg_stub
  MODULE pyxpg
  OUTPUT pyxpg.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:pyxpg>
  DEPENDS pyxpg
)

endif()
