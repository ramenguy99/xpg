// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

#include "common.slang"

// SDF volume.
[vk::binding(0, 0)]
RWTexture3D<float> in_volume;

// Output array of valid blocks. Each element in the array is set to 1 if the block is valid and 0 otherwise.
[vk::binding(1, 0)]
RWStructuredBuffer<uint32_t> out_valid;

struct Constants {
	// Level set for which to check for surface.
    float level;
	uint32_t num_blocks_x;
	uint32_t num_blocks_y;
};

[vk::push_constant]
ConstantBuffer<Constants> g_constants;

groupshared uint s_ballot[MAX_WAVES_IN_GROUP];
groupshared uint s_valid[MAX_WAVES_IN_GROUP];

[shader("compute")]
[numthreads(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z)]
void main(
    uint3  group_id        : SV_GroupID,
    uint3  group_thread_id : SV_GroupThreadID,
	uint   thread_idx      : SV_GroupIndex,
) {
	// Index of the block inside the volume.
	uint3 block_idx = group_id;

	// Index of the cell inside the volume.
    uint3 volume_idx = get_volume_idx(block_idx, group_thread_id);

	// Index of the thread inside the group.

    uint width, height, depth;
    in_volume.GetDimensions(width, height, depth);

	float value;
	uint valid;
	if(all((volume_idx < uint3(width, height, depth)))) {
		// Load volume data if inside the volume.
		value = in_volume.Load(int3(volume_idx));
		valid = 1;
	} else {
		// Set the value to level to ensure that this bit is not set in the ballot.
		value = g_constants.level;
		valid = 0;
	}

	// Count how many invocations are inside the grid.
    uint num_valid = WaveActiveSum(valid);

	// Compute ballot of values below surface level.
	uint count = WaveActiveCountBits(value > g_constants.level);
	if(WaveIsFirstLane()) {
		uint wave_index = thread_idx / WaveGetLaneCount();
		s_ballot[wave_index] = count;
		s_valid[wave_index] = num_valid;
	}

	// Ensure all writes to shared memory are visible.
	GroupMemoryBarrierWithGroupSync();

	// Thread 0 in the block checks if the whole block can be skipped.
	if(thread_idx == 0) {
		// Accumulate ballot results of all threads and total number of threads in the group inside the volume.
		uint total_active = 0;
		uint total_valid = 0;
		for(int i = 0; i < BLOCK_TOTAL_SIZE / WaveGetLaneCount(); i++) {
			total_active += uint(s_ballot[i]);
			total_valid += uint(s_valid[i]);
		}

		// Store if this block intersect the surface.
		// The block is completely inside if all points are inside or if all the valid points
		// are inside or all the valid points are outside.
		uint idx = g_constants.num_blocks_y *  g_constants.num_blocks_x * block_idx.z + g_constants.num_blocks_x * block_idx.y + block_idx.x;
		out_valid[idx] = uint(!(total_active == 0 || total_active == total_valid));
	}
}
