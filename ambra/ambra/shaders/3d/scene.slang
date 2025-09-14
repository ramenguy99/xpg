module scene;

interface ILight
{
    float3 eval(float3 world_position, float3 world_normal);
}

struct PointLight : ILight
{
    float3 position;
    float3 intensity;
    float radius_squared;

    float3 eval(float3 world_position, float3 world_normal)
    {
        float3 dir = position - world_position;
        float d2 = dot(dir, dir);
        float d = sqrt(d2);

        // From "Point Light Attenuation Without Singularity - Cem Yuksel 2020"
        // https://www.cemyuksel.com/research/pointlightattenuation/
        float attenuation = 2 / (d2 + radius_squared + d * sqrt(d2 + radius_squared));

        // Cosine term
        float cosine = max(dot(dir * (1.0 / d), world_normal), 0.0);

        return intensity * cosine * attenuation;
    }
}

struct DirectionalLight : ILight
{
    float3 radiance;
    float3 direction;

    float3 eval(float3 world_position, float3 world_normal)
    {
        return radiance * max(-dot(direction, world_normal), 1.0);
    }
}

struct UniformEnvironmentLight : ILight
{
    float3 radiance;
    float3 eval(float3 world_position, float3 world_normal)
    {
        return radiance;
    }
}

public struct Constants {
    public float4x4 view;
    public float4x4 projection;

    public float3 camera_position;
};

[vk::binding(0, 0)]
public ConstantBuffer<Constants> g_constants;
