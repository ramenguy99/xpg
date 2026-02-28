## Ambra

Ambra is a pure-Python library based on PyXPG for creating 3D visualization and
GUI tools. Ambra aims to be a tool that can be quickly installed in most
environments and can bring any data from any source to the screen in as few
lines of code as possible.

The main motivation for writing Ambra in Python is to minimize the effort and
time required for setup and customizations. Compared to other 3D tools, Ambra is
designed to allow full customization and configuration for the application
needs. You can start using Ambra with some of the default primitives, UIs and
controls, but as your application grows, you are encouraged to customize any big
or small part of the library. For example, you can add your own primitives,
shaders, render passes, data streaming systems, etc..

Even though Python isn't the fastest language, Ambra tries to provide fast
primitives that limit the overhead of the interpreter. This includes supporting
modern GPU features like asynchronous streaming, GPU driven rendering, and
bindless resources.

Ambra has minimal dependencies and should be easy to integrate into any Python
3.8+ environment alongside other packages.

### Quickstart

The easiest way to install Ambra is from prebuilt wheels on Pypi:

```
pip install ambra
```

After the setup you can run the example scripts in `examples/` to
test the installation and learn more about the library.

### Build

Ambra is still in development and changing rapidly, to get the latest version
and for development, it is recommended to build from source both `pyxpg` and
`ambra`:

```
git clone --recursive https://github.com/ramenguy99/xpg.git
cd xpg
pip install scikit-build-core
pip install -C cmake.build-type=Debug -C editable.rebuild=true  --no-build-isolation -ve .
cd ambra
pip install -e .
```

This will install `pyxpg` in debug and editable mode (automatically recompiling
after changes) and `ambra` in editable mode.
