add_executable(sequence
    sequence.cpp
)
target_link_libraries(sequence PRIVATE xpg)
install(TARGETS sequence DESTINATION "bin")
