#!/bin/bash -xe


# pushd build-debug
# make -j 16 pyxpg
# popd

# # Move python 
# cp build-debug/pyxpg*.so python/

# # Generate stubs
# pushd python
# python -m nanobind.stubgen -m pyxpg -r
# popd

# python python/voxels.py

# pushd build-debug
# make -j 16 raytrace
# popd

# # slangc ./shaders/raytrace.comp.slang -o res/raytrace.comp.spirv -target spirv
# ../slang/build-debug/Debug/bin/slangc ./shaders/raytrace.comp.slang -o res/raytrace.comp.spirv -target spirv

# # gdb --args ./build-debug/apps/raytrace/raytrace
# ./build-debug/apps/raytrace/raytrace

pushd build-debug
make -j 16 _pyxpg
popd

# Move python 
cp build-debug/pyxpg*.so python/

# Generate stubs
pushd python
python -m nanobind.stubgen -m pyxpg -r
popd

python python/slang_perf.py
