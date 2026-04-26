@echo off

REM Create working dir
mkdir _gen

REM Generate json metadata from imgui.h
python .\ext\dear_bindings\dear_bindings.py -o _gen\pyxpg .\ext\imgui\imgui.h --nogeneratedefaultargfunctions

REM Before this:
REM - Manually copy ext/implot/implot.h into ext/imgui/implot.h
REM - Fixup implot file:
REM    - Remove deprecate stuff at end of file and version check in the middle
REM    - Remove outdate PlotImage with old imgui texture system
REM    - update ImPlotCond_ to use integer literals
REM    - remove ImPlotSpec templated methods
REM - Copy scripts/implot-header-template.cpp and scripts/implot-header-template.h to ext\dearbindings\src\templates
python .\ext\dear_bindings\dear_bindings.py -o _gen\implot .\ext\imgui\implot.h --nogeneratedefaultargfunctions

REM Generate python bindings and type info
python .\scripts\gen_imgui.py _gen\pyxpg.json
python .\scripts\gen_implot.py _gen\implot.json
