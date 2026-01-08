// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

#ifndef MIN_WAVE_SIZE
#error Missing MIN_WAVE_SIZE define
#define MIN_WAVE_SIZE 32
#endif

#ifndef MAX_WAVE_SIZE
#error Missing MAX_WAVE_SIZE define
#define MAX_WAVE_SIZE 32
#endif

#ifndef BLOCK_SIZE_X
#error Missing BLOCK_SIZE_X define
#define BLOCK_SIZE_X 8
#endif
#ifndef BLOCK_SIZE_Y
#error Missing BLOCK_SIZE_Y define
#define BLOCK_SIZE_Y 8
#endif
#ifndef BLOCK_SIZE_Z
#error Missing BLOCK_SIZE_Z define
#define BLOCK_SIZE_Z 8
#endif

#define BLOCK_TOTAL_SIZE (BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z)

// Map from block index and local cell index in the block to cell index in the volume.
uint3 get_volume_idx(uint3 block_idx, uint3 cell_idx) {
    return block_idx * uint3(BLOCK_SIZE_X - 1, BLOCK_SIZE_Y - 1, BLOCK_SIZE_Z - 1) + cell_idx;
}

#define MAX_WAVES_IN_GROUP ((BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z + MIN_WAVE_SIZE - 1) / MIN_WAVE_SIZE)