import os
import subprocess

shaders_dir = os.path.join(os.path.dirname(__file__), "shaders")
res_dir = os.path.join(os.path.dirname(__file__), "res")

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

    in_path = os.path.join(shaders_dir, f)
    out_path = os.path.join(res_dir, filename + ".spirv")

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