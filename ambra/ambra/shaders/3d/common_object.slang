module common_object;

public struct ObjectConstants {
    public float4x4 transform;
};

[vk::binding(0, 1)]
public ConstantBuffer<ObjectConstants> g_object_constants;
