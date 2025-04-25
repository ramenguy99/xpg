# Roadmap

## Features needed for 0.1.0:

Trivia:
- [ ] Better name
- [ ] Better README
    [ ] Description
    [ ] C++:
        - Build instructions
        - Cmake variables
        - Example apps
    [ ] Python:
        - Pip install instructions
        - Conda install instructions (`conda install conda-forge::vulkan-tools`)

Build:
- [x] Better cmake install with all apps
- [x] Truly optional slang (option does not currently work)
- [x] CI wheels
- [x] Fix static build on windows
- [x] Add scripts / clean imgui bindings gen
    - [x] lz4 is dynamic for some reason -> does not happen in CI, likely my issue with vcpkg lz4
    - [x] Statically link with C++ runtime

Maintenance:
- [x] Update all deps (last: 22/04/2025)
- [ ] Add basic CI build and tests on master or manually triggered (can remove slang to keep it fast, could also be a pipeline option)

C++:
- [x] Cleanup namespaces
- [x] Clean / test (e.g. pull out to a branch)
    - [x] hashmap
    - [x] result
    - [x] framegraph
- [ ] Cleanup apps

Python:
- [x] Check if there is a better way to do imports that works more intuitively
      (likely by importing stuff in __init__.py of subpackage)
- [ ] Hook XPG logs into python logs
    - Two problems with this:
        - Logging is a global concept in XPG, we can use a global callback, but how do we ensure
          this is freed properly? If it's just a module global, what is the best way to expose it?
        - How do we ensure that the logging is freed after everything else? Ideally we would like
          to see teardown logs (we currently don't see the Contxt teardown)
        - Can we somehow bind the lifetime of this to the module? Does not seem to be exposed by
          nanobind, but technically possible also with gc.
- [ ] Low level barriers
- [ ] Queues + queue sync
- [ ] Clean examples
    - [ ] Basic
    - [ ] Gui
    - [ ] Pipeline cache
    - [ ] Sequence (?) -> showcases multithreaded loading
    - [ ] Raytrace
    - [ ] Warp interop
- [ ] Slang:
    - [ ] Pipeline cache with all important inputs
    - [ ] Expose spirv targets
    - [ ] Cleaner handling of multiple entry points
- [ ] Cleanup some stubs with pattern matching file:
    - numpy arrays over buffers -> maybe somehow switch to memory view? should be available everywhere
    - tuple args in window callbacks are actually Tuple[float, float]


## Future

Build:
- [ ] Mac support

Viewer:
- [ ] Primitives
- [ ] Server
- [ ] Gui helpers
- [ ] Viewports
- [ ] Strong focus on extensions

Features (likely at viewer level / helpers):
- [ ] Meshoptimizer + meshlets
- [ ] Gaussian splats
- [ ] Ray marching / octrees
- [ ] Marching cubes
- [ ] Pointclouds
- [ ] Framegraph