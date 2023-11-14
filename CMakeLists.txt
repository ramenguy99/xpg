cmake_minimum_required(VERSION 3.5)

project(XGP VERSION 1.0 LANGUAGES CXX)

add_subdirectory(ext/volk)
add_subdirectory(ext/glfw)
add_subdirectory(ext/glm)

add_executable(app src/main.cpp)
target_link_libraries(app PRIVATE volk_headers glfw glm)

