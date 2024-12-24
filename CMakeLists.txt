cmake_minimum_required(VERSION 3.5)

project(XGP VERSION 1.0 LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 20)

# XPG options
set(XPG_PYTHON ON CACHE BOOL "Build XPG python module")
set(XPG_MSVC_ANALYZE OFF CACHE BOOL "Build XPG python module")

# Configure GLFW
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Build GLFW examples")
set(GLFW_BUILD_TESTS OFF CACHE BOOL "Build GLFW tests")

# Configure ZSTD
set(ZSTD_BUILD_STATIC ON)
set(ZSTD_BUILD_SHARED OFF)
set(ZSTD_BUILD_PROGRAMS OFF)
set(ZSTD_MULTITHREAD_SUPPORT OFF)
set(ZSTD_LEGACY_SUPPORT OFF)

add_subdirectory(ext/glfw)
add_subdirectory(ext/volk)
add_subdirectory(ext/glm)
add_subdirectory(ext/VulkanMemoryAllocator)
add_subdirectory(ext/zstd/build/cmake)

# Compiler specific
if (MSVC AND ${XPG_MSVC_ANALYZE})
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /analyze")
endif()


# add_executable(descs apps/descs.cpp  "lib/imgui_impl.h" "lib/result.h" "lib/graph.h" )
# target_link_libraries(descs PRIVATE volk_headers glfw glm VulkanMemoryAllocator)
# target_include_directories(descs PRIVATE lib shaders ext/imgui ext/atomic_queue/include)

add_executable(bigimage apps/bigimage.cpp "lib/hashmap.h" "lib/bounded_lru_cache.h")
target_link_libraries(bigimage PRIVATE volk_headers glfw glm VulkanMemoryAllocator libzstd_static)
target_include_directories(bigimage PRIVATE lib shaders ext/imgui ext/atomic_queue/include)

# add_executable(sequence apps/sequence.cpp)
# target_link_libraries(sequence PRIVATE volk_headers glfw glm VulkanMemoryAllocator)
# target_include_directories(sequence PRIVATE lib shaders ext/imgui ext/atomic_queue/include)
#
# add_executable(plot apps/plot.cpp)
# target_link_libraries(plot PRIVATE volk_headers glfw glm VulkanMemoryAllocator)
# target_include_directories(plot PRIVATE lib shaders ext/imgui ext/implot ext/atomic_queue/include ext/CLI11)
#
# if(XPG_PYTHON)
#
# # Find python
# if (CMAKE_VERSION VERSION_LESS 3.18)
#   set(DEV_MODULE Development)
# else()
#   set(DEV_MODULE Development.Module)
# endif()
# find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
#
# # Add nanobind
# add_subdirectory(ext/nanobind)
#
# # Create python module
# nanobind_add_module(xpg python/module.cpp)
# target_link_libraries(xpg PRIVATE volk_headers glfw glm VulkanMemoryAllocator)
# target_include_directories(xpg PRIVATE lib shaders ext/imgui ext/implot ext/atomic_queue/include)
#
# endif()
