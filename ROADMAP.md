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
- [ ] Fix / silence warnings 

Maintenance:
- [x] Update all deps (last: 22/04/2025)
- [ ] Add basic CI build and tests on master or manually triggered (can remove slang to keep it fast, could also be a pipeline option)

C++:
- [x] Cleanup namespaces
- [x] Clean / test (e.g. pull out to a branch)
    - [x] hashmap
    - [x] result
    - [x] framegraph
- [ ] Cleanup platform stuff (file IO and threading)
    - maybe pickup a small filesystem library?
    - maybe pickup a small utf8 library too for strings/paths?
- [ ] Cleanup apps
    - [ ] Embed shaders somehow?

Python:
- [x] Check if there is a better way to do imports that works more intuitively
      (likely by importing stuff in __init__.py of subpackage)
- [x] Hook XPG logs into python logs
    - Two problems with this:
        - Logging is a global concept in XPG, we can use a global callback, but how do we ensure
          this is freed properly? If it's just a module global, what is the best way to expose it?
        - How do we ensure that the logging is freed after everything else? Ideally we would like
          to see teardown logs (we currently don't see the Contxt teardown)
        - Can we somehow bind the lifetime of this to the module? Does not seem to be exposed by
          nanobind, but technically possible also with gc.
        - GC on module does not seem to work
        - atexit runs earlier than Context (unless nested in func)
        - user fixes would be:
           - user manually calls cleanup funcs if he wants to see cleanup -> suitable for viewer
           - user manually wraps uses of the library outside global scope -> annoying for small scripts?
             But maybe don't care about cleanup logs?
        - Decided to use global object, if user instantiates it twice it throws, automatically cleaned up,
           can potentially add methods (e.g. log level control). Can be cleaned up before other stuff, but not
           critical.
- [x] Investigate window hang if raise in draw, maybe related to with frame r with command buffer? Fixed, was pipeline cache thread not stopping
- [ ] Finish gfx bindings
    - [x] Image descriptors
    - [x] Sync commands
    - [x] Compute
    - [ ] Queues + queue sync
    - [ ] Barriers
        - [ ] Memory barrier for buffers?
        - [ ] Queue transfer barriers?
        - [ ] Low level combined barrier API?
- [ ] Clean examples
    - [x] Headless graphics and compute
    - [x] Minimal
    - [x] Basic app
    - [-] Voxels
        - Per frame resources
        - Depth buffer
        - Mouse interaction
        - [ ] transform not working properly, z is flipped? Debug this with proper x,y,z axis drawn
    - [-] Raytrace
        - [x] Fix requires spirv1.4
        - [ ] camera controls
        - [ ] basic directional light and brdf
        - [ ] sample accumulation
        - [ ] debug views
    - [ ] Sequence (?) -> showcases multithreaded loading
    - [ ] Warp interop
- [ ] Slang:
    - [x] Compile from string
    - [x] Reflection of resource arrays and maybe other types -> look for descriptor set helper ideas
        - [x] Handling of unbounded descriptors
        - [x] Distinguish SAMPLED_IMAGE vs STORAGE_IMAGE vs COMBINED_IMAGE_SAMPLER
    - [x] Think what makes sense to be hot-reloadable (e.g. does not need python changes to keep working)
            vs what is useful for pipeline instantiation and can be done only once at start.
            [x] maybe add other hooks / callbacks to Pipeline object to make this split more obvious
                init vs create seems useful, even though they both need reflection, can just call it twice at start
            [ ] maybe add helpers that are commonly used in this kind of pipeline creation step
                (keep in mind that often some inputs / logic comes from outside). Do this later with more apps / viewer.
    - [ ] Slang not outputting binding decoration when using parameter block, but reflection seems to get it? Bug in slang?
    - [ ] Fix reflection serialization / deserialization
    - [ ] Expose spirv targets
    - [ ] Pipeline cache with all important inputs
    - [ ] Cleaner handling of multiple entry points
        -> actually not a spirv feature, so can just get rid of this?
        -> should we check that there is only one defined and throw otherwise?
        -> does slang support picking one out of many anyways? maybe we should expose that, e.g. for combined comput / vertex + frag in same source file.
- [x] Cleanup some stubs with pattern matching file:
    - [x] numpy arrays over buffers -> maybe somehow switch to memory view? should be available everywhere
    - [x] tuple args in window callbacks are actually Tuple[float, float]
- [ ] Some validation errors can cause hard segfaults, do we have a way right now
      to catch those and bubble them up to python, or at least print them before segfault?
      e.g. Create image with unsupported format -> Format.R8G8B8 (missing A8) is not supported
        img = Image(ctx, W, H, Format.R8G8B8_UNORM,
                    ImageUsageFlags.COLOR_ATTACHMENT | ImageUsageFlags.TRANSFER_SRC,
                    AllocType.DEVICE)
- [ ] Device features
    - [x] Validation errors when not using vulkan 1.3 (for some reason enabling descriptor indexing is not enough)
    - [ ] Synchronization 2 is not actually optional!
        - Remove this as a device feature, enable if the device supports it, otherwise use the fallback mechanism
          ported / included as the layer mechanism
- [ ] If blocked in process_events -> ctrl+c not working
    - [x] Check if should release GIL
    - [ ] Check if can get interrupt somehow and unblock the loop (e.g. with glfwPostEmptyEvent)
        [x] on windows glfw waits on WaitMessage -> an easy workaraound would be to wait with some timoeout
            and check the signals with PyErr_CheckSignals
            - actually glfw does not seem to tell us if any event was received or the timeout expired, which
               means that we then force a redraw at this timeout which does not seem super ok.
                - can potentially
            - not sure if we can actually install the signal handler here
            -> Using platform specific SetConsoleCtrlHandler works fine
        [ ] check on linux
- [ ] ImGui:
    - [ ] Fix imgui with waitevents on linux, likely need some form of animation frame flag / counter to render at least one additional frame
        - [ ] Also happening on first frame on windows
    - [ ] vec2 / vec4
    - [ ] images interop
    - [ ] fonts
    - [ ] better handling of imgui.ini (default could be autogenerated from filename or smth, definitely should not be the working directory)
    - [ ] Error handling / runtime assertions
- [x] Input callbacks:
    - [x] Add more keys

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