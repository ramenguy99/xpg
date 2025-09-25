#version 450 core

#extension GL_GOOGLE_include_directive : require

layout(location = 0) out vec4 fColor;

layout(push_constant, std430) uniform _pc {
    vec3 color;
} pc;

void main()
{
    fColor = vec4(pc.color.r, pc.color.g, pc.color.b, 1.0);
}
