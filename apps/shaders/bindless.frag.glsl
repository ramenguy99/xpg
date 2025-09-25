#version 450 core

#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

layout(push_constant, std430) uniform _pc {
    uint index;
} pc;

layout(set = 0, binding = 0) readonly buffer Colors {
    vec4 color;
} all_buffers[];


void main()
{
    //out_color = all_buffers[uint(gl_FragCoord.x / 50) % 3 + 1].color;
    
    out_color = all_buffers[pc.index % 4].color;
}
