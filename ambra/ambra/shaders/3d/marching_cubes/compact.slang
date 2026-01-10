// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

#define COMPACT_GROUP_SIZE 128


// Input blocks, valid = 1 means the block is kept and valid = 0 means that the block is discarded.
RWStructuredBuffer<uint32_t> in_valid;

RWStructuredBuffer<uint32_t> out_dispatch_args;
RWStructuredBuffer<uint32_t> out_blocks;

struct Constants {
    // Number of input blocks.
    uint32_t num_blocks;
};

[vk::push_constant]
ConstantBuffer<Constants> g_constants;

[shader("compute")]
[numthreads(COMPACT_GROUP_SIZE, 1, 1)]
void main(
    uint3 thread_id: SV_DispatchThreadID,
) {
    uint thread_idx = thread_id.x;

    // Check if block valid.
    uint valid = 0;
    if(thread_idx < g_constants.num_blocks) {
        valid = in_valid[thread_idx];
    }

    // Exclusive scan of ballot
    uint local_index = WavePrefixCountBits(valid != 0);

    // Highest index of valid lane in the wave
    uint lane_index = WaveGetLaneIndex();
    uint highest = WaveActiveMax(valid != 0 ? lane_index : 0);
    uint global_index = 0;

    // Only the highest valid lane performs the atomic
    if (highest == lane_index)
    {
        // Increment number of blocks for each valid block in the subgroup.
        uint local_size = local_index + (valid != 0);
        InterlockedAdd(out_dispatch_args[0], local_size, global_index);
    }

    // Share global_index across the wave
    global_index = WaveActiveMax(global_index);

    // Write output for valid lanes
    if (valid != 0)
    {
        out_blocks[global_index + local_index] = thread_idx;
    }
}
