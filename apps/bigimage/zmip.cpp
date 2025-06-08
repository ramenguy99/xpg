#include <zstd.h>

#include <xpg/log.h>

#include "zmip.h"

using namespace xpg;

ChunkLoadContext AllocChunkLoadContext(const ZMipFile& zmip) {
    Array<u8> compressed_data(zmip.largest_compressed_chunk_size);
    Array<u8> interleaved(zmip.header.chunk_width * zmip.header.chunk_height * zmip.header.channels);
    Array<u8> deinterleaved(zmip.header.chunk_width * zmip.header.chunk_height * 4);

    ChunkLoadContext result = {
        .zmip = &zmip,
        .compressed_data = move(compressed_data),
        .interleaved = move(interleaved),
        .deinterleaved = move(deinterleaved),
    };
    return result;
}

bool LoadChunk(ChunkLoadContext& load, ChunkId c) {
    const ZMipFile& zmip = *load.zmip;

    usize index = GetChunkIndex(zmip, c);
    usize x = c.x;
    usize y = c.y;
    usize l = c.l;
    ZMipChunk b = zmip.chunks[index];
    if (b.offset + b.size < b.offset) {
        logging::error("bigimage/parse/map", "offset + size overflow on chunk (%zu, %zu) at level %zu", x, y, l);
        return false;
    }
    if (b.offset + b.size > zmip.file.size) {
        logging::error("bigimage/parse/map", "offset + size out of bounds chunk (%zu, %zu) at level %zu", x, y, l);
        return false;
    }

    ArrayView<u8> chunk = load.compressed_data.slice(0, b.size);
    if (platform::ReadAtOffset(zmip.file, chunk, b.offset) != platform::Result::Success) {
        logging::error("bigimage/parse/chunk", "Failed to read %u bytes at offset %zu", b.size, b.offset);
        return false;
    }
    usize frame_size = ZSTD_getFrameContentSize(chunk.data, chunk.length);
    if (frame_size != load.interleaved.length) {
        logging::error("bigimage/parse/chunk", "Compressed chunk frame size %zu does not match expected size %zu", frame_size, load.interleaved.length);
        return false;
    }
    usize zstd_code = ZSTD_decompress(load.interleaved.data, load.interleaved.length, chunk.data, chunk.length);
    if (ZSTD_isError(zstd_code)) {
        logging::error("bigimage/parse/chunk", "ZSTD_decompress failed with error %s (%zu)", ZSTD_getErrorName(zstd_code), zstd_code);
        return false;
    }

    // Undo delta coding
    usize plane_size = (usize)zmip.header.chunk_width * zmip.header.chunk_height;
    for (usize c = 0; c < zmip.header.channels; c++) {
        for (usize i = 1; i < plane_size; i++) {
            load.interleaved[i + plane_size * c] += load.interleaved[i - 1 + plane_size * c];
        }
    }

    // Deinterleave planes and add alpha
    for (usize y = 0; y < zmip.header.chunk_height; y++) {
        for (usize x = 0; x < zmip.header.chunk_width; x++) {
            for (usize c = 0; c < zmip.header.channels; c++) {
                load.deinterleaved[(y * zmip.header.chunk_width + x) * 4 + c] = load.interleaved[((zmip.header.chunk_height * c + y) * zmip.header.chunk_width) + x];
            }
            load.deinterleaved[(y * zmip.header.chunk_width + x) * 4 + 3] = 255;
        }
    }

    return true;
}

ZMipFile LoadZmipFile(const char* path)
{
    platform::File file = {};
    platform::Result r = platform::OpenFile(path, &file);
    if (r != platform::Result::Success) {
        logging::error("bigimage/parse", "Failed to open file");
        exit(102);
    }

    if (file.size < sizeof(ZMipHeader)) {
        logging::error("bigimage/parse", "File smaller than header");
        exit(102);
    }

    ZMipHeader header;
    if (platform::ReadExact(file, BytesOf(&header)) != platform::Result::Success) {
        logging::error("bigimage/parse", "Failed to read header from zmip file");
        exit(102);
    }

    logging::info("bigimage/parse/header", "magic: %zu", header.magic);
    logging::info("bigimage/parse/header", "width: %zu", header.width);
    logging::info("bigimage/parse/header", "height: %zu", header.height);
    logging::info("bigimage/parse/header", "channels: %u", header.channels);
    logging::info("bigimage/parse/header", "chunk_width: %u", header.chunk_width);
    logging::info("bigimage/parse/header", "chunk_height: %u", header.chunk_height);
    logging::info("bigimage/parse/header", "levels: %u", header.levels);

    if (header.channels != 3) {
        logging::error("bigimage/parse", "Currently only 3 channel images are supported, got %u", header.channels);
        exit(102);
    }

    Array<ZMipLevelInfo> levels;
    u32 chunk_offset = 0;
    for (usize l = 0; l < header.levels; l++) {
        usize chunks_x = ((header.width >> l) + header.chunk_width - 1) / header.chunk_width;
        usize chunks_y = ((header.height >> l) + header.chunk_height - 1) / header.chunk_height;
        levels.add({
            .chunks_x = (u32)chunks_x,
            .chunks_y = (u32)chunks_y,
            .offset = chunk_offset
        });
        chunk_offset += (u32)(chunks_x * chunks_y);
    }

    Array<ZMipChunk> chunks(chunk_offset);
    if (platform::ReadExact(file, chunks.as_bytes()) != platform::Result::Success) {
        logging::error("bigimage/parse", "Failed to read chunk data from zmip file");
        exit(102);
    }

    usize largest_compressed_chunk_size = 0;
    for (usize i = 0; i < chunks.length; i++) {
        largest_compressed_chunk_size = Max(largest_compressed_chunk_size, (usize)chunks[i].size);
    }

    ZMipFile zmip = {
        .file = file,
        .header = header,
        .chunks = move(chunks),
        .levels = move(levels),
        .largest_compressed_chunk_size = largest_compressed_chunk_size,
    };

    return zmip;
}
