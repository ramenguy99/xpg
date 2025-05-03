#version 450

struct Camera {
    vec3 position;
    float fov;
};

layout (binding = 0, set = 0) uniform UBO {
    uint width;
    uint height;
    float _padding0;
    float _padding1;
    Camera camera[2];
    mat4x3 transform;
} ubo;

layout (binding = 1, set = 0) uniform UBO2 { 
    Camera camera; 
} ubo2;

layout (binding = 1, set = 4) uniform UBO3 { 
    Camera camera; 
} ubo3;

void main()
{
    gl_Position = vec4(1, 1, 1, 1);
}
