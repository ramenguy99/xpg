module common_object;

public struct ObjectConstants {
    public float3x3 transform;
};

[vk::binding(0, 1)]
public ConstantBuffer<ObjectConstants> g_object_constants;
