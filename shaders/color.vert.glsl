#version 450 core

#extension GL_GOOGLE_include_directive : require

layout (set = 0, binding = 0) uniform u_Constants {
    mat4 u_transform;
};

layout (set = 0, binding = 1) uniform u_Constants3 {
    mat4 u_transform;
} u_array[2];

layout (set = 1, binding = 0) uniform u_Constants2 {
    mat4 u_transform2;
};

layout(location = 0) in vec3 aPos;

out gl_PerVertex {
    vec4 gl_Position;
};

void main()
{
    // gl_Position = u_transform * vec4(aPos, 1.0);
    gl_Position = u_transform * u_transform2 * u_array[0].u_transform * u_array[0].u_transform * vec4(aPos, 1.0);
}
