add_executable(shader_reflection
    shader_reflection.cpp
)

target_link_libraries(shader_reflection PRIVATE xpg slang)

# find_package(slang REQUIRED)
# target_include_directories(shader_reflection PRIVATE ${slang_DIR}/../include)
# target_link_libraries(shader_reflection PRIVATE xpg slang::slang)

# target_compile_definitions(shader_reflection PRIVATE SLANG_STATIC)
# target_include_directories(shader_reflection PRIVATE ${slang_DIR}/include)
# target_link_directories(shader_reflection PRIVATE ${slang_DIR}/lib)
# target_link_libraries(shader_reflection PRIVATE xpg
#     compiler-core
#     core
#     slang-cpp-parser
#     slang-rt
#     slang
#     miniz
#     lz4
# )