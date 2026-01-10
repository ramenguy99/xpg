// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

#include "common.slang"
#include "tables.slang"

[vk::binding(0, 0)]
RWTexture3D<float> in_volume;

[vk::binding(1, 0)]
StructuredBuffer<uint32_t> in_blocks;

[vk::binding(2, 0)]
StructuredBuffer<uint32_t> in_tri_table;

[vk::binding(3, 0)]
RWStructuredBuffer<float> out_positions;

[vk::binding(4, 0)]
RWStructuredBuffer<float> out_normals;

[vk::binding(5, 0)]
RWStructuredBuffer<uint32_t> out_indices;

[vk::binding(6, 0)]
RWStructuredBuffer<uint32_t> out_num_vertices;

[vk::binding(7, 0)]
RWStructuredBuffer<uint32_t> out_num_indices;

struct Constants {
    // Level set for which to check for surface.
    float level;

    // Number of blocks in the X directions.
    uint32_t num_blocks_x;

    // Number of blocks in the Y direction.
    uint32_t num_blocks_y;

    // Boolean flag for inverting normals.
    bool invert_normals;

    // Spacing of volume cells.
    float3 volume_spacing;

    // Maximum number of vertices.
    uint max_vertices;

    // Maximum number of triangle indices.
    uint max_indices;
};

[vk::push_constant]
ConstantBuffer<Constants> g_constants;

// OPT: Can reduce shared memory usage by overlapping some of these arrays that are not used at the same time.
groupshared float s_volume_data[BLOCK_SIZE_Z + 1][BLOCK_SIZE_Y + 1][BLOCK_SIZE_X + 1];

#define MAX(a, b) ((a) > (b) ? (a) : (b))
groupshared uint s_scan_vertices[MAX(MAX_WAVES_IN_GROUP, MAX_WAVE_SIZE)];
groupshared uint s_scan_triangles[MAX(MAX_WAVES_IN_GROUP, MAX_WAVE_SIZE)];

groupshared uint s_vertices_base;
groupshared uint s_triangles_base;

#ifndef COUNT_ONLY
groupshared float3 s_gradient[BLOCK_SIZE_Z][BLOCK_SIZE_Y][BLOCK_SIZE_X];

groupshared uint s_scan_block[BLOCK_SIZE_Z][BLOCK_SIZE_Y][BLOCK_SIZE_X];
#endif

// Lookup block index in volume from the compacted index.
uint3 getBlockIdx(uint idx) {
    uint block = in_blocks[idx];
    uint3 block_idx;
    block_idx.x = block % g_constants.num_blocks_x;
    block_idx.y = (block % (g_constants.num_blocks_x * g_constants.num_blocks_y)) / g_constants.num_blocks_x;
    block_idx.z = block / (g_constants.num_blocks_x * g_constants.num_blocks_y);
    return block_idx;
}

#ifndef COUNT_ONLY
// Return t such that a * (1 - t) + b * t = v
float inverse_lerp(float v, float a, float b) {
    return (v - a) / (b - a);
}

// Lookup vertex index from neighbouring cell in shared memory.
uint get_vertex_index(uint3 cell_idx, uint cube_index, uint tri_index) {
    uint edge = in_tri_table[cube_index * 16 + tri_index];
    uint4 edge_map = table_edge_mapping[edge];

    uint bitmask = s_scan_block[cell_idx.z + edge_map.z][cell_idx.y + edge_map.y][cell_idx.x + edge_map.x];
    uint n_offset = bitmask & 0x1FFF;

    // OPT: This can be optimized by storing a mask in w and using a popcnt or a lookup table.
    // (can also maybe expand the lookup table for edge_mapping with 3 extra bits?)
    uint n_index = 0;
    for(int j = 0; j < edge_map.w; j++) {
        n_index += (bitmask >> (13 + j)) & 1;
    }

    return n_index + n_offset;
}

// Lookup gradients from neighbouring cells and interpolate them.
float3 getNormal(uint3 idx_a, uint3 idx_b, float t) {
    float3 a = s_gradient[idx_a.z][idx_a.y][idx_a.x];
    float3 b = s_gradient[idx_b.z][idx_b.y][idx_b.x];
    float3 n = normalize(lerp(a, b, t));

    if(g_constants.invert_normals) {
        n = -n;
    }

    return n;
}
#endif

[shader("compute")]
[numthreads(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z)]
void main(
    uint3 group_id: SV_GroupID,
    uint3 cell_idx: SV_GroupThreadID,
    uint thread_idx: SV_GroupIndex,
) {
    // Index of the block inside the volume.
    uint3 block_idx = getBlockIdx(group_id.x);

    // Index of the cell inside the volume.
    uint3 volume_idx = get_volume_idx(block_idx, cell_idx);

    uint width, height, depth;
    in_volume.GetDimensions(width, height, depth);
    uint3 volume_size = uint3(width, height, depth);

    float value;
    if(all(volume_idx < volume_size)) {
        // Load volume data if inside the volume.
        value = in_volume[volume_idx];
    } else {
        value = g_constants.level;
    }

    // Store sdf value of this thread in shared memory.
    s_volume_data[cell_idx.z][cell_idx.y][cell_idx.x] = value;

    // Ensure writes to s_volume_data are visible.
    GroupMemoryBarrierWithGroupSync();

    // Extract sdf values from neighbours for vertices of this cell.
    float f000 = s_volume_data[cell_idx.z    ][cell_idx.y    ][cell_idx.x    ];
    float f001 = s_volume_data[cell_idx.z    ][cell_idx.y    ][cell_idx.x + 1];
    float f010 = s_volume_data[cell_idx.z    ][cell_idx.y + 1][cell_idx.x    ];
    float f011 = s_volume_data[cell_idx.z    ][cell_idx.y + 1][cell_idx.x + 1];
    float f100 = s_volume_data[cell_idx.z + 1][cell_idx.y    ][cell_idx.x    ];
    float f101 = s_volume_data[cell_idx.z + 1][cell_idx.y    ][cell_idx.x + 1];
    float f110 = s_volume_data[cell_idx.z + 1][cell_idx.y + 1][cell_idx.x    ];
    float f111 = s_volume_data[cell_idx.z + 1][cell_idx.y + 1][cell_idx.x  +1];

    // Check if border cells.
    bool3 cell_border_low = cell_idx == uint3(0);
    bool3 cell_border_high = cell_idx == uint3(BLOCK_SIZE_X - 1, BLOCK_SIZE_Y - 1, BLOCK_SIZE_Z - 1);

    // Check if at volume border or outside.
    bool3 volume_border_low = volume_idx == uint3(0);
    bool3 volume_border_high = volume_idx >= volume_size - uint3(1);

#ifndef COUNT_ONLY
    // Lookup neighbours for computing gradient with finite differences.
    float f00n, f0n0, fn00;

    // Lower neighbours, can be in shared memory or have to be looked up.
    if(cell_border_low.x && !volume_border_low.x) {
        f00n = in_volume[uint3(volume_idx.x - 1, volume_idx.y, volume_idx.z)];
    } else {
        f00n = s_volume_data[cell_idx.z][cell_idx.y][cell_idx.x - 1];
    }
    if(cell_border_low.y && !volume_border_low.y) {
        f0n0 = in_volume[uint3(volume_idx.x, volume_idx.y - 1, volume_idx.z)];
    } else {
        f0n0 = s_volume_data[cell_idx.z][cell_idx.y - 1][cell_idx.x];
    }
    if(cell_border_low.z && !volume_border_low.z) {
        fn00 = in_volume[uint3(volume_idx.x, volume_idx.y, volume_idx.z - 1)];
    } else {
        fn00 = s_volume_data[cell_idx.z - 1][cell_idx.y][cell_idx.x];
    }

    // Higher neighbours have to be looked up if at cell border.
    if(cell_border_high.x && !volume_border_high.x) {
        f001 = in_volume[uint3(volume_idx.x + 1, volume_idx.y, volume_idx.z)];
    }
    if(cell_border_high.y && !volume_border_high.y) {
        f010 = in_volume[uint3(volume_idx.x, volume_idx.y + 1, volume_idx.z)];
    }
    if(cell_border_high.z && !volume_border_high.z) {
        f100 = in_volume[uint3(volume_idx.x, volume_idx.y, volume_idx.z + 1)];
    }


    // Compute gradient with central differences for internal cells
    // and using forward/backward differences at volume boundaries.
    float3 g;
    if(volume_border_high.x) {
        g.x = (f000 - f00n) * g_constants.volume_spacing.x * 2.0;
    } else if(volume_border_low.x) {
        g.x = (f001 - f000) * g_constants.volume_spacing.x * 2.0;
    } else {
        g.x = (f001 - f00n) * g_constants.volume_spacing.x;
    }

    if(volume_border_high.y) {
        g.y = (f000 - f0n0) * g_constants.volume_spacing.y * 2.0;
    } else if(volume_border_low.y) {
        g.y = (f010 - f000) * g_constants.volume_spacing.y * 2.0;
    } else {
        g.y = (f010 - f0n0) * g_constants.volume_spacing.y;
    }

    if(volume_border_high.z) {
        g.z = (f000 - fn00) * g_constants.volume_spacing.z * 2.0;
    } else if(volume_border_low.z) {
        g.z = (f100 - f000) * g_constants.volume_spacing.z * 2.0;
    } else {
        g.z = (f100 - fn00) * g_constants.volume_spacing.z;
    }

    // Store gradient in shared memory.
    s_gradient[cell_idx.z][cell_idx.y][cell_idx.x] = g;
#endif

    // Classify cell comparing with surface level value.
    uint field[8];
    field[0] = uint(f000 > g_constants.level);
    field[1] = uint(f001 > g_constants.level);
    field[2] = uint(f011 > g_constants.level);
    field[3] = uint(f010 > g_constants.level);
    field[4] = uint(f100 > g_constants.level);
    field[5] = uint(f101 > g_constants.level);
    field[6] = uint(f111 > g_constants.level);
    field[7] = uint(f110 > g_constants.level);

    // Pack booleans into an 8 bit integer for the lookup table.
    uint cube_index =
        field[0] << 0 |
        field[1] << 1 |
        field[2] << 2 |
        field[3] << 3 |
        field[4] << 4 |
        field[5] << 5 |
        field[6] << 6 |
        field[7] << 7;

    // Compute which edges of this cell generate a vertex.
    uint3 edge_index;
    edge_index.x = field[0] ^ field[1];
    edge_index.y = field[0] ^ field[3];
    edge_index.z = field[0] ^ field[4];

    // Discard vertices that go outside the cell or outside the volume.
    bool3 border = cell_border_high || volume_border_high;

    if(border.x) {
        edge_index.x = 0;
    }

    if(border.y) {
        edge_index.y = 0;
    }

    if(border.z) {
        edge_index.z = 0;
    }

    // Compute number of vertices and triangles required by this cell.
    uint num_vertices = edge_index.x + edge_index.y + edge_index.z;
    uint num_triangles = 0;

    // Border cells might generate vertices but always generate 0 triangles.
    if(!any(border)) {
        // OPT: store this table already divided by 3.
        num_triangles = table_num_vertices[cube_index] / 3;
    }


    // Perform per-block exclusive scan of number of vertices and number of triangles
    // that will be created by this block.
    uint wave_index = thread_idx / WaveGetLaneCount();
    uint wave_count = BLOCK_TOTAL_SIZE / WaveGetLaneCount();

    // Per subgroup scan.
    uint vertex_index = WavePrefixSum(num_vertices);
    uint triangle_index = WavePrefixSum(num_triangles);

    // Store highest value for each subgroup in shared memory.
    uint lane_index = WaveGetLaneIndex();
    uint highest_in_subgroup = WaveActiveMax(lane_index);
    if(lane_index == highest_in_subgroup) {
        s_scan_vertices[wave_index] = vertex_index + num_vertices;
        s_scan_triangles[wave_index] = triangle_index + num_triangles;
    }

    GroupMemoryBarrierWithGroupSync();

    // Subgroup 0 performs inclusive scan across the top values of each subgroup.
    if(wave_index == 0) {
        // Since our block size is 512 this loops only if 512 / subgroupSize > subgroupSize
        // thus if subgroupSize ** 2 < 512 <=> subgroupSize < 32
        uint prev_sum_vertices = 0;
        uint prev_sum_triangles = 0;

        for (uint i = 0; i < wave_count; i += WaveGetLaneCount()) {
            uint sum_vertices = 0;
            uint sum_triangles = 0;
            if(i + lane_index < wave_count) {
                sum_vertices = s_scan_vertices[i + lane_index];
                sum_triangles = s_scan_triangles[i + lane_index];
            }

            uint exclusive_sum_vertices = WavePrefixSum(sum_vertices) + prev_sum_vertices;
            s_scan_vertices[i + lane_index] = exclusive_sum_vertices;

            uint exclusive_sum_triangles = WavePrefixSum(sum_triangles)+ prev_sum_triangles;
            s_scan_triangles[i + lane_index] = exclusive_sum_triangles;

            prev_sum_vertices = WaveActiveMax(exclusive_sum_vertices + sum_vertices);
            prev_sum_triangles = WaveActiveMax(exclusive_sum_triangles + sum_triangles);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // Add scan of previous subgroup to vertex and triangle index.
    vertex_index += s_scan_vertices[wave_index];
    triangle_index += s_scan_triangles[wave_index];

    // Last thread in the block increments atomic counters for triangles and vertices;
    if (thread_idx == (BLOCK_TOTAL_SIZE - 1)) {
        uint vertices_base = 0;
        uint triangles_base = 0;
        InterlockedAdd(out_num_vertices[0], vertex_index + num_vertices, vertices_base);
        InterlockedAdd(out_num_indices[0], (triangle_index + num_triangles) * 3, triangles_base);
        s_vertices_base = vertices_base;
        s_triangles_base = triangles_base;
    }

#ifndef COUNT_ONLY
    // Write out vertex information for other threads in the group.
    s_scan_block[cell_idx.z][cell_idx.y][cell_idx.x] = (edge_index.z << 15) | (edge_index.y << 14) | (edge_index.x << 13) | vertex_index;

    // Ensure writes to s_scan_block, s_vertices_base and s_triangles_base are visible to the rest of the group.
    GroupMemoryBarrierWithGroupSync();

    // Compute offsets into output arrays.
    uint vertices_base = s_vertices_base;
    uint triangles_base = s_triangles_base;

    uint global_vertex_index = vertices_base + vertex_index;
    uint global_triangle_index = triangles_base + triangle_index * 3;

    // Return early if out of bounds in the vertices or indices array.
    if(global_vertex_index + num_vertices > g_constants.max_vertices ||
       global_triangle_index + num_triangles * 3 > g_constants.max_indices) {
        return;
    }

    // Output vertices.
    if(edge_index.x > 0) {
        float t = inverse_lerp(g_constants.level, f000, f001);

        float3 v = float3(volume_idx.x + t, volume_idx.y, volume_idx.z) * g_constants.volume_spacing;
        out_positions[global_vertex_index * 3 + 0] = v.x;
        out_positions[global_vertex_index * 3 + 1] = v.y;
        out_positions[global_vertex_index * 3 + 2] = v.z;

        float3 n = getNormal(cell_idx, cell_idx + uint3(1, 0, 0), t);
        out_normals[global_vertex_index * 3 + 0] = n.x;
        out_normals[global_vertex_index * 3 + 1] = n.y;
        out_normals[global_vertex_index * 3 + 2] = n.z;

        global_vertex_index += 1;
    }

    if(edge_index.y > 0) {
        float t = inverse_lerp(g_constants.level, f000, f010);

        float3 v = float3(volume_idx.x, volume_idx.y + t, volume_idx.z) * g_constants.volume_spacing;
        out_positions[global_vertex_index * 3 + 0] = v.x;
        out_positions[global_vertex_index * 3 + 1] = v.y;
        out_positions[global_vertex_index * 3 + 2] = v.z;

        float3 n = getNormal(cell_idx, cell_idx + uint3(0, 1, 0), t);
        out_normals[global_vertex_index * 3 + 0] = n.x;
        out_normals[global_vertex_index * 3 + 1] = n.y;
        out_normals[global_vertex_index * 3 + 2] = n.z;

        global_vertex_index += 1;
    }

    if(edge_index.z > 0) {
        float t = inverse_lerp(g_constants.level, f000, f100);

        float3 v = float3(volume_idx.x, volume_idx.y, volume_idx.z + t) * g_constants.volume_spacing;
        out_positions[global_vertex_index * 3 + 0] = v.x;
        out_positions[global_vertex_index * 3 + 1] = v.y;
        out_positions[global_vertex_index * 3 + 2] = v.z;

        float3 n = getNormal(cell_idx, cell_idx + uint3(0, 0, 1), t);
        out_normals[global_vertex_index * 3 + 0] = n.x;
        out_normals[global_vertex_index * 3 + 1] = n.y;
        out_normals[global_vertex_index * 3 + 2] = n.z;
    }

    // Output triangles.
    for(uint i = 0; i < num_triangles; i++) {
        uint i0 = vertices_base + get_vertex_index(cell_idx, cube_index, i * 3 + 0);
        uint i1 = vertices_base + get_vertex_index(cell_idx, cube_index, i * 3 + 1);
        uint i2 = vertices_base + get_vertex_index(cell_idx, cube_index, i * 3 + 2);

        out_indices[global_triangle_index + 0] = i0;
        out_indices[global_triangle_index + 1] = i1;
        out_indices[global_triangle_index + 2] = i2;
        global_triangle_index += 3;
    }
#endif
}
