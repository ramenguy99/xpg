// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

module scene;

public struct Constants {
    public float4x4 view;
    public float4x4 projection;

    public float3 world_camera_position;

    public float3 ambient_light;
    public uint32_t has_environment_light;

    public float2 focal;
    public float2 inverse_viewport_size;

    public float max_specular_mip;
    public uint32_t directional_lights_count;
};

[vk::binding(0, 0)]
public ConstantBuffer<Constants> g_constants;

[vk::binding(1, 0)]
public StructuredBuffer<uint8_t> g_lights;
