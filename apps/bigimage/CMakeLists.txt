add_executable(bigimage 
    bigimage.cpp
    zmip.cpp
    zmip.h
    chunk_cache.cpp
    chunk_cache.h
)
target_link_libraries(bigimage PRIVATE xpg libzstd_static)
install(TARGETS bigimage DESTINATION "bin")