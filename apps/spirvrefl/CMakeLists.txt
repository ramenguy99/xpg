add_executable(spirvrefl
    spirvrefl.cpp
)

target_link_libraries(spirvrefl PRIVATE xpg spirv-cross-reflect spirv-cross-core glslang glslang-default-resource-limits)