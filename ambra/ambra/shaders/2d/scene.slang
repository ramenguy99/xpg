module scene;

public struct Constants {
    public float4x4 view;
    public float4x4 projection;

    public float3 camera_position;
    public uint32_t num_direcrional_lights;
};

[vk::binding(0, 0)]
public ConstantBuffer<Constants> g_constants;

[vk::binding(1, 0)]
public StructuredBuffer<uint8_t> g_lights;
