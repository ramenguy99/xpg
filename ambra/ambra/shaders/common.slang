module common;

public struct Constants {
    public float4x4 view;
    public float4x4 projection;

    public float3 camera_position;
};

[vk::binding(0, 0)]
public ConstantBuffer<Constants> g_constants;