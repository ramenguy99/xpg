add_subdirectory(bigimage)
add_subdirectory(descs)
add_subdirectory(sequence)
add_subdirectory(sync)
add_subdirectory(plot)
add_subdirectory(scene3d)
add_subdirectory(raytrace)
# add_subdirectory(minimal)

if (XPG_BUILD_GLSLANG_AND_SPIRV)
    add_subdirectory(spirvrefl)
endif()

if (XPG_BUILD_SLANG)
    # Only available if building slang
    add_subdirectory(shader_reflection)
endif()