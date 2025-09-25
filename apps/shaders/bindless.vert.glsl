#version 450 core

layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec2 a_uv;

out gl_PerVertex {
    vec4 gl_Position;
};

layout(location = 0) out vec2 out_uv;

void main()
{
    gl_Position = vec4(a_pos, 1.0);
    out_uv = a_uv;
}
