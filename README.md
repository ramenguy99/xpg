# XPG

XPG is a library for writing native cross-platform real-time graphics
applications and tools with minimal dependencies.  It is based on Vulkan for
performance and portability. It provides helpers to simplify application
development but exposes Vulkan concepts, functions and types directly to give
full control to applications and allow access the all functionality and
extensions that Vulkan offers.

XPG is written in a very C-like like C++, it makes lightweight use of templates
and destructors for a few simple data structures and it uses operator
overloading for bounds checks and vector math.

## Features

- Helpers for device initialization, (optionally) window management and most of
  the Vulkan API.
- ImGui integration based on the GLFW and Vulkan backends.
- Shader compilation and reflection utilities based on Slang or glslang and
  SPIR-V tools.
- First-class python bindings for graphics, imgui and shader compilation
  and reflection.

## Project structure

The project currently contains 3 main components:

- **XPG**: C++ library. Source is in `src/include/` and `src/lib/`. Example
  applications are in `apps/` and shaders in `shaders/`.
- **PyXPG**: Python bindings for XPG using nanobind. Source of the bindings is
  in `src/python/` and examples are in `python/`.
- **Ambra**: Ambra is a pure-python 3D viewer based on PyXPG. Source is in
  `ambra/` and examples in `ambra/examples/`. Ambra is meant as a standalone
  tool and is one of the main motivations driving the design of both XPG and
  PyXPG.

## Dependencies

All dependencies are included as git submodules for versioning and for
self-contained builds and source distributions:

- Vulkan SDK components (tracking the latest official release).
- GLFW as for cross-platform window creation and input collection.
- ImGui and ImPlot for UI and dear_bindings for imgui python binding generation.
- glslang and slang for shader compilation and reflection.
- GLM for vector and matrix math.
- Nanobind for python bindings.

On Linux, GLFW has a few build dependencies for X11 and Wayland support.
The easiest way to install these is from your distribution package manager. See
the Build section below for more details.

## Build

Building the XPG library, bindings and sample applications requires a C++20
compiler and cmake. The recommended way to build XPG applications is to build
XPG as a static library, this will produce portable executables with no runtime
dependencies other than the system standard libraries.

XPG can be built for any platform supported by the Vulkan SDK and
GLFW. It's currently tested on x64 Windows, x64 and aarch64 Linux and aarch64
MacOS.

Building XPG does not require the Vulkan SDK to be installed. But, if available,
XPG can enable and configure the validation layer at runtime.

### CMake options

The CMake build can be configured with the following XPG-specific options:

- `XPG_BUILD_APPS`: If set to `ON` build example XPG applications in `./app`.
  Defualt: `ON`.
- `XPG_BUILD_PYTHON`: If set to `ON` build the `pyxpg` python module.
  Requires python header files to be installed and discoverable by
  `find_package`. Default: `ON`.
- `XPG_BUILD_SLANG`: If set to `ON` build slang and the `pyxpg.slang` python
  module. Default: `ON`.
- `XPG_BUILD_GLSLANG_AND_SPIRV`: If set to `ON` build glslang and spirv tools
  (experimental). Deafult: `OFF`.
- `XPG_MSVC_ANALYZE`: If set to `ON` and compiling with MSVC, build with the
  `/analyze` flag set for additional checks (significantly increases
  compilation time). Default: `OFF`.

Most dependencies also use CMake and their variables can also be specified at
configuration time. By default, XPG configures some of the dependencies to
disable optional features and to configure them for static linking.

### Windows

A C++ compiler and cmake can be installed with the C/C++ development package
from the Visual Studio installer. XPG compilation on Windows is tested with both
MSVC and Clang.  For building PyXPG Python must be installed system-wide along
with development header files.

### Linux

XPG is tested with both Clang and GCC on x64 and aarch64 Ubuntu 24.04. Other
distributions are expected to work by installing the respective packages.
XPG builds GLFW with support for X11, Wayland or both depending on which
packages are found at CMake configuration time.

The following packages are required for building XPG with X11 support:
```
sudo apt install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
```

The following packages are required for building XPG with Wayland support:
```
sudo apt install wayland-dev libxkbcommon-dev wayland-protocols
```

The following packages are required for building PyXPG:
```
sudo apt install libpython3-dev
```

Python wheels are built on `musllinux_1_2` and `manylinux_2_28`. See
`pyproject.toml` for the list of build dependencies on those platforms.

### MacOS

On MacOS XPG is tested using Clang on aarch64 MacOS 14. Intel MacOS should
also work but is not tested.

On MacOS XPG requires MoltenVK for Vulkan support. By default, MoltenVK is
expected to be installed by the Vulkan SDK and dynamically loaded at runtime.
Alternatively, MoltenVK can be linked statically to simplify distribution of
binaries by specifying the `XPG_MOLTENVK_STATIC_DIR` CMake option (??).

### Example build commands

#### Full build

#### Minimal build (no python bindings and no slang)

#### Editable python build

## Ambra

