struct MeshInstance {
    float4x4 transform;

    uint vertex_offset;
    uint index_offset;
    uint _padding0;
    uint _padding1;

    uint albedo_index;
    uint normal_index;
    uint specular_index;
    uint emissive_index;

    float4 albedo_value;
    float4 specular_value;
    float4 emissive_value;
};

struct Camera {
    float3 position;
    uint _padding;

    float3 direction;
    float film_dist;
};

struct Constants {
    uint width;
    uint height;
    uint _padding0;
    uint _padding1;

    Camera camera;
};

StructuredBuffer<float3, ScalarDataLayout> normals_buffer;
StructuredBuffer<float2> uvs_buffer;
StructuredBuffer<uint> indices_buffer;
StructuredBuffer<MeshInstance> instances_buffer;
RaytracingAccelerationStructure acceleration_structure;
SamplerState sampler;
[vk::image_format("rgba32f")]
RWTexture2D<float3> output;
Texture2D<float4> textures[];

struct Frame {
    ConstantBuffer<Constants> constants;
};
ParameterBlock<Frame> frame;


inline void GenerateCameraRay(uint2 p, out float3 origin, out float3 direction)
{
    uint2 size = uint2(frame.constants.width, frame.constants.height);

    float2 xy = p;
    float2 offset = xy / size.xy * 2.0 - 1.0;

    float3 camera_forward = frame.constants.camera.direction;
    float3 world_up = float3(0, 0, 1);
    float3 camera_right = normalize(cross(camera_forward, world_up));
    float3 camera_up = cross(camera_right, camera_forward);
    float3 camera_p = frame.constants.camera.position;

    float film_dist = frame.constants.camera.film_dist;
    float2 film_size = float2(1.0, 1.0);
    film_size.y = (float)size.y / (float)size.x;
    float2 half_film = film_size * 0.5f;
    float3 film_center = camera_p - film_dist * camera_forward;

    origin = film_center + offset.x * half_film.x * camera_right
                         + offset.y * half_film.y * camera_up;

    direction = normalize(camera_p - origin);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 thread_idx : SV_DispatchThreadID)
{
    uint2 p = thread_idx.xy;
    if(any(p >= uint2(frame.constants.width, frame.constants.height))) {
        return;
    }

    float3 dir;
    float3 origin;
    GenerateCameraRay(p, origin, dir);

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.001;
    ray.TMax = 100000.0;

    RayQuery<RAY_FLAG_NONE> q;
    let rayFlags = RAY_FLAG_NONE;
    q.TraceRayInline(acceleration_structure, rayFlags, 0xff, ray);

    q.Proceed();
    if(q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        float t = q.CommittedRayT();

        // Hit position
        float3 position = origin + t * dir;

        // Mesh info
        uint mesh_index = q.CommittedRayInstanceId();
        MeshInstance instance = instances_buffer[mesh_index];

        // Primitive info
        uint triangle_index = q.CommittedRayPrimitiveIndex();
        uint index_offset = instance.index_offset;
        uint vertex_offset = instance.vertex_offset;

        uint3 indices;
        indices.x = indices_buffer[triangle_index * 3 + 0 + index_offset];
        indices.y = indices_buffer[triangle_index * 3 + 1 + index_offset];
        indices.z = indices_buffer[triangle_index * 3 + 2 + index_offset];

        float2 barycentrics = q.CommittedRayBarycentrics();

        float3 normal =
            normals_buffer[indices.x + vertex_offset] * (1 - barycentrics.x - barycentrics.y) +
            normals_buffer[indices.y + vertex_offset] * barycentrics.x +
            normals_buffer[indices.z + vertex_offset] * barycentrics.y;
        float3 N = normalize(mul((float3x3)q.CandidateObjectToWorld3x4(), normal));

        float2 uv =
            uvs_buffer[indices.x + vertex_offset] * (1 - barycentrics.x - barycentrics.y) +
            uvs_buffer[indices.y + vertex_offset] * barycentrics.x +
            uvs_buffer[indices.z + vertex_offset] * barycentrics.y;

        float3 color;
        if (instance.albedo_index != ~0) {
            color = textures[NonUniformResourceIndex(instance.albedo_index)].SampleLevel(sampler, uv, 0).rgb;
        } else {
            color = instance.albedo_value.rgb;
        }
        // output[thread_idx.xy] = (float)index_offset / 8487678;
        // output[thread_idx.xy] = (float)vertex_offset / 2937174;
        // output[thread_idx.xy] = (float)mesh_index / 1520;
        // output[thread_idx.xy] = float3(uv * 0.5 + 0.5, 0.0f);
        // output[thread_idx.xy] = N * 0.5 + 0.5;
        // output[thread_idx.xy] = (float)instance.albedo_index / 200;
        output[thread_idx.xy] = color;
    } else {
        if (bool(((thread_idx.x ^ thread_idx.y) >> 4) & 1)) {
            output[thread_idx.xy] = float3(0.1, 0.1, 0.1);
        } else {
            output[thread_idx.xy] = float3(0.0, 0.0, 0.0);
        }
    }
}
