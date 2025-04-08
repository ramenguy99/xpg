cmake_minimum_required(VERSION 3.18)

project(XGP VERSION 1.0 LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 20)

# XPG options
set(XPG_BUILD_APPS ON CACHE BOOL "Build XPG applications")
set(XPG_BUILD_PYTHON ON CACHE BOOL "Build XPG python module")
set(XPG_BUILD_SLANG ON CACHE BOOL "Build XPG with slang")
set(XPG_MSVC_ANALYZE OFF CACHE BOOL "Build with /analyze when using MSVC (heavily increases build times)")

# Configure GLFW
if (SKBUILD)
    set(GLFW_INSTALL OFF)
endif ()
set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)

# Configure Volk
set(VOLK_PULL_IN_VULKAN OFF)
set(VOLK_HEADERS_ONLY ON)

# Configure ZSTD
set(ZSTD_BUILD_STATIC ON)
set(ZSTD_BUILD_SHARED OFF)
set(ZSTD_BUILD_PROGRAMS OFF)
set(ZSTD_MULTITHREAD_SUPPORT OFF)
set(ZSTD_LEGACY_SUPPORT OFF)

# Configure slang
if (XPG_BUILD_SLANG)
    set(SLANG_ENABLE_CUDA OFF)
    set(SLANG_ENABLE_OPTIX OFF)
    set(SLANG_ENABLE_NVAPI OFF)
    set(SLANG_ENABLE_XLIB OFF)
    set(SLANG_ENABLE_SLANG_RHI OFF)
    set(SLANG_ENABLE_DXIL OFF)
    set(SLANG_ENABLE_GFX OFF)
    set(SLANG_ENABLE_SLANGD OFF)
    set(SLANG_ENABLE_SLANGC OFF)
    set(SLANG_ENABLE_SLANGRT OFF)
    set(SLANG_ENABLE_SLANG_GLSLANG OFF)
    set(SLANG_ENABLE_TESTS OFF)
    set(SLANG_ENABLE_EXAMPLES OFF)
    set(SLANG_ENABLE_REPLAYER OFF)
    set(SLANG_ENABLE_SPLIT_DEBUG_INFO OFF)
    set(SLANG_EMBED_STDLIB_SOURCE OFF)
    set(SLANG_EMBED_STDLIB ON)
    set(SLANG_LIB_TYPE STATIC)
    set(SLANG_SLANG_LLVM_FLAVOR DISABLE)
    add_subdirectory(ext/slang EXCLUDE_FROM_ALL)
endif()

# Add external dependencies
add_subdirectory(ext/glfw)
add_subdirectory(ext/Vulkan-Headers)
add_subdirectory(ext/volk)
add_subdirectory(ext/glm)
add_subdirectory(ext/VulkanMemoryAllocator)
add_subdirectory(ext/zstd/build/cmake EXCLUDE_FROM_ALL)

# Compiler specific
if (MSVC AND ${XPG_MSVC_ANALYZE})
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /analyze")
endif()

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-volatile -Wno-nullability-completeness")
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
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>/ext/Vulkan-Utility-Libraries/include
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(xpg PUBLIC Vulkan-Headers volk_headers glfw glm VulkanMemoryAllocator)

if (NOT SKBUILD)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/include/xpg DESTINATION "include" FILES_MATCHING PATTERN "*.h")
    install(TARGETS xpg LIBRARY DESTINATION "lib")
endif()

if (MSVC OR CMAKE_C_SIMULATE_ID STREQUAL "MSVC")
    target_compile_definitions(xpg PUBLIC _CRT_SECURE_NO_WARNINGS)
endif()


# Add apps
if (XPG_BUILD_APPS)
    add_subdirectory(apps)
endif()


# Python module
if(XPG_BUILD_PYTHON)
    set(PYXPG_SOURCES
        src/python/module.cpp
        src/python/py_gfx.cpp
        src/python/py_imgui.cpp
    )
    if(XPG_BUILD_SLANG)
        set(PYXPG_SOURCES ${PYXPG_SOURCES}
        src/python/py_slang.cpp
        )
    endif()

    # Find python
    find_package(Python 3.8
        REQUIRED COMPONENTS Interpreter Development.Module
        OPTIONAL_COMPONENTS Development.SABIModule)

    # Add nanobind
    add_subdirectory(ext/nanobind)

    # Create python module
    nanobind_add_module(_pyxpg
        STABLE_ABI
        ${PYXPG_SOURCES}
    )
    if(XPG_BUILD_SLANG)
        target_compile_definitions(_pyxpg PRIVATE PYXPG_SLANG_ENABLED=1)
        target_link_libraries(_pyxpg PRIVATE xpg slang)
    else()
        target_compile_definitions(_pyxpg PRIVATE PYXPG_SLANG_ENABLED=0)
        target_link_libraries(_pyxpg PRIVATE xpg)
    endif()

    # set_target_properties(pyxpg PROPERTIES OUTPUT_NAME "xpg")
    nanobind_add_stub(pyxpg_stub
        MODULE _pyxpg
        OUTPUT pyxpg/__init__.pyi
        MARKER_FILE pyxpg/py.typed
        PYTHON_PATH $<TARGET_FILE_DIR:_pyxpg>
        DEPENDS _pyxpg
    )

    install(TARGETS _pyxpg DESTINATION "pyxpg")

    # Build time generated files for typing
    install(FILES 
        ${CMAKE_CURRENT_BINARY_DIR}/pyxpg/__init__.pyi
        ${CMAKE_CURRENT_BINARY_DIR}/pyxpg/py.typed
        DESTINATION
        "pyxpg")

    # Python sources
    install(FILES 
        ${CMAKE_SOURCE_DIR}/src/python/pyxpg/__init__.py
        DESTINATION
        "pyxpg")

    install(FILES 
        ${CMAKE_SOURCE_DIR}/src/python/pyxpg/imgui/__init__.py
        ${CMAKE_SOURCE_DIR}/src/python/pyxpg/imgui/__init__.pyi
        ${CMAKE_SOURCE_DIR}/src/python/pyxpg/imgui/py.typed
        DESTINATION
        "pyxpg/imgui")
    
    # Slang
    if(XPG_BUILD_SLANG)
        nanobind_add_stub(pyxpg_slang_stub
            MODULE _pyxpg.slang
            OUTPUT pyxpg/slang/__init__.pyi
            MARKER_FILE pyxpg/slang/py.typed
            PYTHON_PATH $<TARGET_FILE_DIR:_pyxpg>
            DEPENDS _pyxpg
        )

        install(FILES 
            ${CMAKE_CURRENT_BINARY_DIR}/pyxpg/slang/__init__.pyi
            ${CMAKE_CURRENT_BINARY_DIR}/pyxpg/slang/py.typed
            DESTINATION
            "pyxpg/slang")

        install(FILES 
            ${CMAKE_SOURCE_DIR}/src/python/pyxpg/slang/__init__.py
            DESTINATION
            "pyxpg/slang")
    endif()

endif()
