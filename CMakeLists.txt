cmake_minimum_required(VERSION 3.5)

project(XGP VERSION 1.0 LANGUAGES CXX)

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Build GLFW examples")
set(GLFW_BUILD_TESTS OFF CACHE BOOL "Build GLFW tests")

add_subdirectory(ext/glfw)
add_subdirectory(ext/volk)
add_subdirectory(ext/glm)
add_subdirectory(ext/VulkanMemoryAllocator)

add_executable(sequence apps/sequence.cpp)
target_link_libraries(sequence PRIVATE volk_headers glfw glm VulkanMemoryAllocator)
target_include_directories(sequence PRIVATE lib shaders ext/imgui ext/atomic_queue/include)

add_executable(plot apps/plot.cpp)
target_link_libraries(plot PRIVATE volk_headers glfw glm VulkanMemoryAllocator)
target_include_directories(plot PRIVATE lib shaders ext/imgui ext/implot ext/atomic_queue/include ext/CLI11)