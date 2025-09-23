# XPG

XPG is a library for writing native cross-platform real-time graphics
applications and tools with minimal dependencies. It is based on Vulkan for
performance and portability. It provides helpers to simplify application
development but exposes Vulkan concepts, functions, and types directly to give
full control to applications and allow access all functionality and extensions
that Vulkan offers.

XPG is written in a very C-like like C++, it makes lightweight use of templates
and destructors for a few simple data structures, and it uses operator
overloading for bounds checks and vector math.

## Features

- Helpers for device initialization, (optionally) window management, and most of
  the Vulkan API.
- ImGui integration based on the GLFW and Vulkan backends.
- Shader compilation and reflection utilities based on Slang or glslang and
  SPIR-V tools.
- First-class Python bindings for graphics, imgui and shader compilation
  and reflection for Python 3.8+.

## Project structure

The project currently contains 3 main components:

- **XPG**: C++ library. Source is in `src/include/` and `src/lib/`. Example
  applications are in `apps/` and shaders in `shaders/`.
- **PyXPG**: Python bindings for XPG using nanobind. Source of the bindings is
  in `src/python/` and examples are in `python/`.
- **Ambra**: Ambra is a pure-Python 3D viewer based on PyXPG. Source is in
  `ambra/` and examples in `ambra/examples/`. Ambra is meant as a standalone
  tool and is one of the main motivations driving the design of both XPG and
  PyXPG.

## Dependencies

All dependencies are included as git submodules for versioning and for
self-contained builds and source distributions:

- Vulkan SDK components (tracking the latest official release).
- GLFW for cross-platform window creation and input collection.
- ImGui and ImPlot for UI and dear_bindings for imgui Python binding generation.
- glslang and slang for shader compilation and reflection.
- GLM for vector and matrix math.
- Nanobind for writing Python bindings.

On Linux, GLFW has a few build dependencies for X11 and Wayland support.
The easiest way to install these is from your distribution package manager. See
the Build section below for more details.

## Build

Building the XPG library, Python bindings, and sample applications requires a
C++20 compiler and cmake. The recommended way to build XPG applications is to
build XPG as a static library, this will produce portable executables with no
runtime dependencies other than the system standard libraries.

XPG can be built for any platform supported by the Vulkan SDK and
GLFW. It's currently tested on x64 Windows, x64 and aarch64 Linux and aarch64
MacOS.

Building XPG does not require the Vulkan SDK to be installed. But, if available,
XPG can enable and configure the validation layer at runtime.

### CMake options

The CMake build can be configured with the following XPG-specific options:

- `XPG_BUILD_APPS`: If set to `ON` build example XPG applications in `./app`.
  Default: `ON`.
- `XPG_BUILD_PYTHON`: If set to `ON` build the `pyxpg` Python module.
  Requires Python header files to be installed and discoverable by
  `find_package`. Default: `ON`.
- `XPG_BUILD_SLANG`: If set to `ON` build slang and the `pyxpg.slang` Python
  module. Default: `ON`.
- `XPG_BUILD_GLSLANG_AND_SPIRV`: If set to `ON` build glslang and spirv tools
  (experimental). Deafult: `OFF`.
- `XPG_MSVC_ANALYZE`: If set to `ON` and compiling with MSVC, build with the
  `/analyze` flag set for additional checks (significantly increases
  compilation time). Default: `OFF`.
- `XPG_MOLTENVK_PATH`: If specified, link statically with the MoltenVK library
  pointed to this path. Otherwise Vulkan is loaded dynamically at runtime by
  volk.

Most dependencies also use CMake and their variables can also be specified at
configuration time. By default, XPG configures some of the dependencies to
disable optional features and configure them for static linking.

### Windows

A C++ compiler and cmake can be installed with the C/C++ development package
from the Visual Studio installer. XPG compilation on Windows is tested with both
MSVC and Clang.  For building PyXPG Python must be installed system-wide along
with development header files.

### Linux

XPG is tested with both Clang and GCC on x64 and aarch64 Ubuntu 24.04. Other
distributions are expected to work by installing the respective packages.
XPG builds GLFW with support for X11, Wayland, or both, depending on which
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

PyXPG Python wheels for distribution are built on `manylinux_2_28` and
`musllinux_1_2`. See `pyproject.toml` for the list of build dependencies on
those platforms.

### MacOS

On MacOS XPG is tested using Clang on aarch64 MacOS 14. Intel MacOS should
also work but is not regularly tested.

MoltenVK is required for Vulkan support. By default, MoltenVK is expected to be
installed by the Vulkan SDK and dynamically loaded at runtime by volk.
Alternatively, MoltenVK can be linked statically to simplify distribution of
binaries by specifying the `XPG_MOLTENVK_PATH` CMake option.

### Example build commands

#### Full build

```
cmake -B build/
cmake --build build/
```

#### Minimal build (no apps, no Python bindings, and no slang)

```
cmake -B build/ -DXPG_BUILD_APPS=OFF -DXPG_BUILD_PYTHON=OFF -DXPG_BUILD_SLANG=OFF
cmake --build build/
```

#### Editable Python build

For development of PyXPG it can be convenient to setup an editable build that
will check for changes and recompile the bindings whenever the `pyxpg` module is
imported.

This can be achieved with the following command:
```
pip install scikit-build-core
pip install -C cmake.build-type=Debug -C editable.rebuild=true  --no-build-isolation -ve .
```

See the [scikit-build-core documentation](https://scikit-build-core.readthedocs.io/en/latest/configuration/index.html#configuration)
for more details on build configuration.

## Ambra

Ambra is a pure-Python 3D viewer based on PyXPG. The goal of Ambra is to be an
easy-to-use and easy-to-customize library for creating 3D visualization and GUI
tools. Ambra is a tool that can be quickly installed in most environments and
can bring any data from any source to the screen in as few lines of code as
possible.

The main motivation for writing Ambra in Python is to minimize the effort and
time required for setup and customizations. Compared to other 3D tools, Ambra is
designed to allow full customization and configuration for the application
needs. You can start using Ambra with some of the default primitives, UIs and
controls, but as your application grows, you are encouraged to customize any big
or small part of the library. For example, you can add your own primitives,
shaders, render passes, data streaming systems, etc..

Even though Python isn't the fastest language, Ambra tries to provide fast
primitives that limit the overhead of the interpreter. This includes exposing
asynchronous data streaming, GPU driven rendering, and APIs that operate on
batches of elements.

Ambra has minimal dependencies and should be easy to integrate into any Python
3.8+ environment alongside other packages.

### Setup

Ambra is deep in development and Python wheels are not yet available. Currently
`ambra` does not depend explicitly on `pyxpg` in its `pyproject.toml` to avoid
downloading wheels from Pypi when installed.

The recommended way to install Ambra is to clone the repository and install from
source both `pyxpg` and `ambra` to ensure the latest version of both is in use.

Example setup:
```
git clone --recursive https://github.com/ramenguy99/xpg.git
cd xpg
pip install scikit-build-core
pip install -C cmake.build-type=Debug -C editable.rebuild=true  --no-build-isolation -ve .
cd ambra
pip install -e .
```

This will install `pyxpg` in debug and editable mode and `ambra` in editable
mode. After the setup you can run the example scripts in `ambra/examples/` to
test the installation.
