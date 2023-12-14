cmake_minimum_required(VERSION 3.5)

project(XGP VERSION 1.0 LANGUAGES CXX)

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Build GLFW examples")
set(GLFW_BUILD_TESTS OFF CACHE BOOL "Build GLFW tests")

add_subdirectory(ext/glfw)
add_subdirectory(ext/volk)
add_subdirectory(ext/glm)
add_subdirectory(ext/VulkanMemoryAllocator)

add_executable(app src/main.cpp "src/defines.h" "src/threading.h" "src/platform.h" "src/gfx.h" "src/buffered_stream.h")
target_link_libraries(app PRIVATE volk_headers glfw glm VulkanMemoryAllocator)
target_include_directories(app PRIVATE ext/imgui ext/atomic_queue/include shaders)

