#version 450 core

#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) in vec3 aPos;

out gl_PerVertex {
    vec4 gl_Position;
};

layout(location = 0) out vec3 world_position;

void main()
{
    world_position = aPos;
    gl_Position = u_constants.transform * vec4(aPos, 1.0);
}
