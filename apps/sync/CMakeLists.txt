add_executable(sync
    sync.cpp
)
target_link_libraries(sync PRIVATE xpg)
install(TARGETS sync DESTINATION "bin")
