bool traceRayNearestHit(
    RaytracingAccelerationStructure sceneBVH,
    float3 rayOrigin,
    float3 rayDir,
    out float t,
    out int primitiveIndex)
{
    RayDesc ray;
    ray.Origin = rayOrigin;
    ray.TMin = 0.01f;
    ray.Direction = rayDir;
    ray.TMax = 1e4f;
    RayQuery<RAY_FLAG_NONE> q;
    let rayFlags = RAY_FLAG_NONE;

    q.TraceRayInline(
        sceneBVH,
        rayFlags,
        0xff,
        ray);

    q.Proceed();
    if(q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        t = q.CommittedRayT();
        primitiveIndex = q.CommittedPrimitiveIndex();
        return true;
    }
    t = 0;
    primitiveIndex = 0;
    return false;
}

[vk::binding(0, 0)]
RaytracingAccelerationStructure acceleration_structure;

[vk::binding(1, 0)]
RWTexture2D<float3> resultTexture;

struct Constants {
    uint width;
    uint height;
};

[vk::binding(2, 0)]
ConstantBuffer<Constants> constants;

[shader("compute")]
[numthreads(32,32,1)]
void computeMain(uint3 thread_idx : SV_DispatchThreadID)
{
    if (thread_idx.x >= (int)constants.width) return;
    if (thread_idx.y >= (int)constants.height) return;

    // float frameWidth = uniforms.screenWidth / uniforms.screenHeight * uniforms.frameHeight;
    // float imageY = (threadIdx.y / uniforms.screenHeight - 0.5f) * uniforms.frameHeight;
    // float imageX = (threadIdx.x / uniforms.screenWidth - 0.5f) * frameWidth;
    // float imageZ = uniforms.focalLength;
    // float3 rayDir = normalize(uniforms.cameraDir.xyz*imageZ - uniforms.cameraUp.xyz * imageY + uniforms.cameraRight.xyz * imageX);
    // float4 resultColor = 0;

    // int primitiveIndex;
    // float intersectionT;
    // if (traceRayNearestHit(sceneBVH, uniforms.cameraPosition.xyz, rayDir, intersectionT, primitiveIndex))
    // {
    // }
    float2 uv = ((thread_idx.xy + 0.5) / float2(constants.width, constants.height)) * 2.0 - 1.0;

    float t = 0;
    uint index = 0;
    if (traceRayNearestHit(acceleration_structure, float3(uv, -1.0), float3(0, 0, 1), t, index)) {
        resultTexture[thread_idx.xy] = float3(1, 0, 0);
    } else {
        if (bool(((thread_idx.x ^ thread_idx.y) >> 4) & 1)) {
            resultTexture[thread_idx.xy] = float3(0.1, 0.1, 0.1);
        }
    }
}
