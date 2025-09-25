#version 450 core

#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) in vec3 world_position;

layout(location = 0) out vec4 fColor;

void main()
{
    vec3 L = normalize(vec3(-1, -1, 1));

    vec3 t = dFdx(world_position);
    vec3 b = dFdy(world_position);
    vec3 N = normalize(cross(t, b));

    float kA = 0.3;
    float kD = max(0.0, dot(N, L));

    vec3 color = u_constants.color * min(kA + kD, 1.0);
    fColor = vec4(color, 1.0);
}
