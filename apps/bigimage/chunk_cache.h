#include <xpg/gfx.h>
#include <xpg/pool.h>
#include <xpg/threading.h>

#include "zmip.h"

// LRU cache of image chunks:
// - Cache entries are in 3 possible states:
//   - Empty: (nothing has ever been loaded on this entry, it's free for use)                                                            | refcount = 0, entry = null
//   - In use: (some drawcall is in flight that uses this entry)                                                                         | refcount > 0, entry = valid
//   - Not in use: (some drawcall previously referenced this entry, it still contains it's data but it is available for reuse if needed) | refcount = 0, entry = valid
// - The cache is managed by the main thread with this cycle:
//   - START:  At start entries are initialized and vulkan images are allocated, the entries start in the Empty state
//   - RESIZE: We can allocated additional blocks to the cache, entry pointers stay valid because the PooledQueue ensures pointers are stable.
//   - Main loop:
//       - Take the chunks that were used when this frame was in flight the last time and decrement the refcount of all the references.
//         If any goes to zero put this cache entry in the "Not In Use" state. This should is a queue of entries from which we can remove from the middle.
//       - Take chunks that are needed for this frames and increment refcount of all the references.
//         If the chunk is already in use (refcount > 0) we are done.
//         If the chunk is not already in use:
//            If the chunk still points to a valid cache entry, remove it from the free list.
//            If the chunk cache entry is not valid grab a new one from the empty list and fill it (and invalidate the previous chunk that referred to it, if any).
struct ChunkCache {
    // Entry in the cache
    static constexpr usize INVALID_CHUNK = ~(usize)(0);
    struct Entry {
        usize chunk_index;
        u32 image_index;
    };

    // Information about a chunk, contains all possible chunks
    struct Chunk {
        u32 refcount;                                 // Frames in flight that use this chunk
        PoolQueue<Entry>::Entry* lru_entry;     // Optional entry into the LRU cache. The value in the entry indexes both the images array and descriptor set.
    };

    // Struct passed to worker threads to load a chunk
    struct Work {
        ArrayView<u8> buffer_map;
        ChunkId c;
        BlockingCounter* work_done_counter;
    };

    ChunkCache(const ZMipFile& zmip, usize cache_size, usize upload_buffers_count, usize num_workers, usize num_frames_in_flight,
        const gfx::Context& vk, const gfx::BindlessDescriptorSet bindless);

    void destroy_resources(const gfx::Context& vk);
    void resize(usize cache_size, usize upload_buffers_count, const gfx::Context& vk, const gfx::BindlessDescriptorSet bindless);
    void release_chunk(usize chunk_index);
    static void worker_func(WorkerPool::WorkerInfo* worker_info, void* user_data);
    void request_chunk_batch(ArrayView<ChunkId> chunk_ids, ArrayView<u32> output_descriptors, const gfx::Context& vk, const gfx::BindlessDescriptorSet& bindless, VkCommandBuffer cmd, u32 frame_index);
    u32 request_chunk_sync(ChunkId c, const gfx::Context& vk, const gfx::BindlessDescriptorSet& bindless);

    inline Chunk& get_chunk(ChunkId c) {
        usize index = GetChunkIndex(zmip, c);
        return chunks[index];
    }

    // ZMip file
    const ZMipFile& zmip;

    // Chunks metadata
    Array<Chunk> chunks;

    // LRU cache of images
    Array<gfx::Image> images;
    PoolQueue<Entry> lru;

    // Sync upload
    ChunkLoadContext sync_load_context;

    // Async upload
    ObjArray<Array<gfx::Buffer>> upload_buffers; // Pool of staging buffers for upload, one pool for each frame.
    ObjArray<ChunkLoadContext> worker_infos;     // Worker data, one per worker thread
    Array<Work> work_items;                      // Work data, one per work item
    WorkerPool worker_pool;
    BlockingCounter work_done_counter;

    // Stats
    u64 chunk_memory_size;
};
