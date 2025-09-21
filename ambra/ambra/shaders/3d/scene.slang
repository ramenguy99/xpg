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

public struct DirectionalLight : ILight
{
    public float4x4 orthographic_camera;
    public float3 radiance;
    public uint32_t shadowmap_index;
    public float3 direction;
    public float bias;

    public float3 eval(float3 world_position, float3 world_normal)
    {
        float3 L = radiance * max(-dot(direction, world_normal), 0.0);

        if (shadowmap_index != 0xffffffff) {
            // Compute fragment coordinates in light space
            float3 light_pos = mul(orthographic_camera, float4(world_position, 1.0)).xyz;

            // Assume not in shadow if outside frustum in z direction
            if (light_pos.z < 1.0)
            {
                // Compute value to compared with taking bias into account
                float z = light_pos.z;
                float b = max(bias * 0.5 * (1.0 + dot(direction, world_normal)), bias);
                float cmp_value = z - b;

                // Sample shadowmap
                float shadow = g_directional_shadowmaps[shadowmap_index].SampleCmpLevelZero(g_directional_shadowmap_sampler, light_pos.xy * 0.5 + 0.5, cmp_value);

                // Weight radiance by shadow term
                L *= shadow;
            }
        }

        return L;
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
    public uint32_t directional_lights_count;
};

[vk::binding(0, 0)]
public ConstantBuffer<Constants> g_constants;

[vk::binding(1, 0)]
public StructuredBuffer<DirectionalLight> g_directional_lights;

[vk::binding(2, 0)]
public Texture2D g_directional_shadowmaps[];

[vk::binding(3, 0)]
public SamplerComparisonState g_directional_shadowmap_sampler;
