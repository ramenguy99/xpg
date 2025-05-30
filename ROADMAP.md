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
- [ ] Fix / silence warnings (first enable all useful ones)
    - [ ] MSVC
    - [ ] clang on windows
    - [ ] gcc on linux
    - [ ] clang on linux

Maintenance:
- [x] Update all deps (last: 22/04/2025)
- [ ] Add basic CI build and tests on master or manually triggered (can remove slang to keep it fast, could also be a pipeline option)

C++:
- [x] Cleanup namespaces
- [x] Clean / test (e.g. pull out to a branch)
    - [x] hashmap
    - [x] result
    - [x] framegraph
- [x] Check support for present modes
- [ ] Cleanup platform stuff (file IO and threading)
    - maybe pickup a small filesystem library?
    - maybe pickup a small utf8 library too for strings/paths?
- [ ] Cleanup apps
    - [ ] Embed shaders somehow?
    - [ ] Run with syncrhonization validation and GPU based validation
        - [ ] Completely switch to syncrhonization 2 for submission? Probably need to fix barriers for submit and present at COLOR_ATTACHMENT_OUTPUT stage

Python:
- [x] Check if there is a better way to do imports that works more intuitively
      (likely by importing stuff in __init__.py of subpackage)
- [x] Hook XPG logs into python logs
- [x] Investigate window hang if raise in draw, maybe related to with frame r with command buffer? Fixed, was pipeline cache thread not stopping
- [ ] Finish gfx bindings
    - [x] Image descriptors
    - [x] Sync commands
    - [x] Compute
    - [x] Queues + queue sync
    - [ ] Barriers
        - [x] Memory barrier for buffers?
        - [x] Queue transfer barriers
        - [ ] Low level combined barrier API?
    - [x] Helpers for buffer upload with fallback, think about differnt allocation use cases
        Preferred solutions:
        - One time upload at start:
            - if device mapped available use it
            - if device mapped not available upload with sync queue
        - Per frame upload:
            - if device mapped available:
                - if small use it
                - if large frame queue + barriers (requires some helper funcs stuff to do the barrier if needed)
            - if device mapped unavailable:
                - if super small can use system memory and rely on caching
                - if large use frame gfx queue
        - What APIs do we expose to make the simple case easy?
        - If the upload is performance critical we can have an helper that the user
        can check if device mapped memory is available and then decide for himself.
        - Maybe also think about helpers for copy queue at the same time
        - Plan:
            [x] Make from_data use DEVICE_MAPPED with fallback and alloc staging buffer + sync copy transfer if needed
            [x] Keep API as it is, create helpers for common use cases
                [x] Small data upload (constants) -> device mapped or fallback to upload.
                [x] Large upload (streaming) -> If integrated, device mapped or host mapped. If device + staging buffer with upload.
    - [x] Cleanup command buffers, queues, sync commands and frame API. See what is the shared functionality and if we can improve this a bit.
- [ ] Clean examples
    - [x] Headless graphics and compute
    - [x] Minimal
    - [x] Basic app
    - [-] Voxels
        - [x] Per frame resources
        - [x] Depth buffer
        - [x] Mouse interaction
        - [ ] transform not working properly, z is flipped, likely vulkan viewport? Debug this with proper x,y,z axis drawn
    - [-] Raytrace
        - [x] Fix requires spirv1.4
        - [x] Efficient image upload with preallocated batch -> can probably do this from python with sync queue
        - [ ] camera controls
        - [ ] basic directional light and brdf
        - [ ] sample accumulation
        - [ ] debug views
    - [x] Sequence
        - [x] Sync loading
        - [x] Async disk loading
        - [x] Buffered stream is actually flawed when doing GPU copies, buffers cannot be immediately replaced
              when finished using them on the CPU. Also need to synchronize with the GPU. I think something like
              an LRU cache of buffers actually makes a lot of sense then. Similar to what we used for bigimaage.
              The idea is that we will grab buffers from the LRU in the uploader thread. This will block until
              a free buffer is ready and will acquire it. The main thread will submit prefetch requests and
              wait for the current frame to be ready. Once ready it will be transitioned to in use.
              If the worker threads fall behind, the main thread will block due to the buffer not being ready.
              This in turn will prevent submission of more work. E.g. there will be at most BUFFER_COUNT buffers
              in flight at any point in time.
              Let's think at the same time about frame pacing / async upload.
              See docs for pipelining samples.
              Most promising solution seems to be:
              - fully asynchronous data loading, only bound by number of buffers free, with prefetch logic
              - synchronous upload on copy queue if available, wait for CPU buffer, submit on other queue, otherwise on any other async queue (async compute, other separate graphics queue)
              - release buffers synchronously when done with frame -> after waiting for fence and knowing what we will neeed this frame and in future
        - [x] Async upload with copy queue
        - [x] Keyboard input
        - [x] Handle throws in threadpool jobs
        - [x] Integrated GPU does not have transfer queue, therefore prefers using CPU buffers directly. Use physical device type to switch strategy.
        - [ ] Stop pre-fetching if we detect skipping for most frames? Keep skip statistics?
    - [ ] Warp interop
        - [ ] Requires instructions to build warp from our branch
- [x] Slang:
    - [x] Compile from string
    - [x] Reflection of resource arrays and maybe other types -> look for descriptor set helper ideas
        - [x] Handling of unbounded descriptors
        - [x] Distinguish SAMPLED_IMAGE vs STORAGE_IMAGE vs COMBINED_IMAGE_SAMPLER
    - [x] Think what makes sense to be hot-reloadable (e.g. does not need python changes to keep working)
            vs what is useful for pipeline instantiation and can be done only once at start.
            [x] maybe add other hooks / callbacks to Pipeline object to make this split more obvious
                init vs create seems useful, even though they both need reflection, can just call it twice at start
            [x] maybe add helpers that are commonly used in this kind of pipeline creation step
                (keep in mind that often some inputs / logic comes from outside). Do this later with more apps / viewer.
    - [x] Fix reflection serialization / deserialization
    - [-] Slang not outputting binding decoration when using parameter block, but reflection seems to get it? Bug in slang?
          -> opened issue on github, seems to be in general related to directly nesting Resources in ParameterBlock, works fine
             with direct data and structs in between
          [-] Should fix the ParameterBlock with data directly to create implicit constant buffer for completeness.
    - [x] Expose spirv targets (does slang increase the target for us automatically if we just say spirv? maybe thats better?) -> no, this is now exposed
    - [x] Pipeline cache with all important inputs
    - [x] Cleaner handling of multiple entry points
        - slang supports picking the entry point you want but always generates one with main now.
        - output spirv always uses "main", we go for single spirv entry point for each shader
        -> potentially can support multiple spirv entry points in a single spirv module in the future?
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
    - [ ] Would be nice to have optional features and check if they are supported later. Not clear what's easiest way to do this.
          And how to handle priorities / scores.
        - [ ] Make use of this to use some fallback when vk_khr_timestamp_calibration is not available
- [x] If blocked in process_events -> ctrl+c not working
    - [x] Check if should release GIL
    - [x] Check if can get interrupt somehow and unblock the loop (e.g. with glfwPostEmptyEvent)
        - [x] on windows glfw waits on WaitMessage -> an easy workaraound would be to wait with some timoeout
            and check the signals with PyErr_CheckSignals
            - actually glfw does not seem to tell us if any event was received or the timeout expired, which
               means that we then force a redraw at this timeout which does not seem super ok.
                - can potentially
            - not sure if we can actually install the signal handler here
            -> Using platform specific SetConsoleCtrlHandler works fine
        - [x] check on linux -> already works
- [ ] ImGui:
    - [x] vec2 / vec4
    - [x] Drawlist
    - [ ] Remove text_end (or other string end type of pointers from APIs)
    - [ ] Fix begin returns tuple, many examples expect bool
    - [ ] Fix imgui with waitevents on linux, likely need some form of animation frame flag / counter to render at least one additional frame
        - [ ] Also happening on first frame on windows
    - [ ] images interop
    - [ ] fonts
    - [ ] better handling of imgui.ini (default could be autogenerated from filename or smth, definitely should not be the working directory)
    - [ ] Can we turn error handling / runtime assertions into exceptions?
- [x] Input callbacks:
    - [x] Add more keys
- [ ] None converts to a nullptr nb::ref, makes a lot of our code potentially segfault
    - Opened discussion in nanobind repo
    - Only applies to containers, can check those manually and throw for now probably
    -> we have some potentially difficult to debug segfaults but can live with this for now

## Future

Build:
- [ ] Mac support

Docs:
- [ ] Doc comments and documentation website

Python:
- [ ] glslang bindings for compiling and reflection
    - [ ] fix slang build when using this
- [ ] Tracy module built-in into xpg. Repackage their bindings for CPU stuff, expose vulkan API tracing, and add compat bindings with our GPU stuff.
- [ ] Expose host image copy and timeline semaphores?
    -> timeline semaphores should be avilable everywhere. Ideally subclass / parameter of Semaphore and transparent to queue waits but with extra APIs on the object.
    -> Host image copy can be used automatically for Image.with_data() to or manually with exposed host operations. Not available on AMD

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