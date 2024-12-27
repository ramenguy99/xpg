#pragma once

#include <xpg/array.h>
#include <xpg/platform.h>

#pragma pack(push, 1)
struct ZMipHeader {
    u64 magic;
    u64 width;
    u64 height;
    u32 channels;
    u32 chunk_width;
    u32 chunk_height;
    u32 levels;
};
struct ZMipChunk {
    u64 offset;
    u32 size;
};
#pragma pack(pop)

struct ZMipLevelInfo {
    u32 chunks_x;
    u32 chunks_y;
    u32 offset;
};

struct ZMipFile {
    platform::File file;
    ZMipHeader header;
    Array<ZMipChunk> chunks;
    Array<ZMipLevelInfo> levels;
    usize largest_compressed_chunk_size;
};

struct ChunkId {
    u32 x, y, l;

    ChunkId(u32 x, u32 y, u32 l) : x(x), y(y), l(l) {}
};

struct ChunkLoadContext {
    const ZMipFile* zmip;
    Array<u8> compressed_data;
    Array<u8> interleaved;
    Array<u8> deinterleaved;
};

ChunkLoadContext AllocChunkLoadContext(const ZMipFile& zmip);
bool LoadChunk(ChunkLoadContext & load, ChunkId c);
ZMipFile LoadZmipFile(const char* path);

inline usize GetChunkIndex(const ZMipFile& zmip, ChunkId c) {
    ZMipLevelInfo level = zmip.levels[c.l];
    return level.offset + (usize)c.y * level.chunks_x + c.x;
}

