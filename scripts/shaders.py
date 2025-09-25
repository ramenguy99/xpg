import os
import subprocess
from pathlib import Path

shaders_dir = Path(__file__).parent.parent.joinpath("apps", "shaders")
res_dir = Path(__file__).parent.joinpath("res")

kinds = {
    'vert',
    'frag',
    'comp',
}

for f in os.listdir(shaders_dir):
    filename, ext = os.path.splitext(f)

    components = filename.split(".")
    if len(components) < 2:
        continue

    name, kind = components[-2:]
    if kind not in kinds:
        continue

    in_path = shaders_dir.joinpath(f)
    out_path = res_dir.joinpath(filename + ".spirv")

    glsl_args = [
        "glslangValidator",
        "-V",
        in_path,
        "-o",
        out_path
    ]
    subprocess.run(glsl_args)

# Slang:
# C:\VulkanSDK\1.3.296.0\Bin\slangc.exe .\shaders\bigimage.comp.slang -o res\bigimage.comp.spirv -target spirv
