#!/bin/bash -xe

# Create working dir
mkdir -p _gen

# Generate json metadata from imgui.h
python ./ext/dear_bindings/dear_bindings.py -o _gen/pyxpg ./ext/imgui/imgui.h --nogeneratedefaultargfunctions

# Generate python bindings and type info
python ./scripts/gen_imgui.py _gen/pyxpg.json