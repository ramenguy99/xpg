import os
import subprocess

shaders_dir = os.path.join(os.path.dirname(__file__), "shaders")
res_dir = os.path.join(os.path.dirname(__file__), "res")

extensions = {
    '.vert',
    '.frag',
    '.comp',
}

for f in os.listdir(shaders_dir):
    if os.path.splitext(f)[1] not in extensions:
        continue

    in_path = os.path.join(shaders_dir, f)
    out_path = os.path.join(res_dir, f + ".spirv")

    glsl_args = [
        "glslangValidator",
        "-V",
        in_path,
        "-o",
        out_path
    ]
    subprocess.run(glsl_args)