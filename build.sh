#!/bin/bash -xe

pushd build-debug
make -j 16 pyxpg
popd

# Move python 
cp build-debug/pyxpg*.so python/

# Generate stubs
pushd python
python -m nanobind.stubgen -m pyxpg -r
popd

python python/voxels.py