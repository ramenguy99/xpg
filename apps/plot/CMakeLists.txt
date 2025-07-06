add_executable(plot
    roboto-medium.h
    plot.cpp
)
target_link_libraries(plot PRIVATE xpg)
target_include_directories(plot PRIVATE ../../ext/implot ../../ext/CLI11)
install(TARGETS plot DESTINATION "bin")
