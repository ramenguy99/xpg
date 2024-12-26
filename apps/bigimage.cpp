#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utility> // std::move
#include <functional> // std::function
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <semaphore.h>
#include <pthread.h>
#endif
#define VOLK_IMPLEMENTATION
#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>

#define VMA_STATIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#define _GLFW_VULKAN_STATIC
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#endif
#include <GLFW/glfw3.h>

#undef APIENTRY
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui.h>
#include <imgui.cpp>
#include <imgui_demo.cpp>
#include <imgui_draw.cpp>
#include <imgui_tables.cpp>
#include <imgui_widgets.cpp>

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_glfw.cpp>

#undef VK_NO_PROTOTYPES
#include <backends/imgui_impl_vulkan.h>
#include <backends/imgui_impl_vulkan.cpp>

#include <atomic_queue/atomic_queue.h>

#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <zstd.h>

#define XPG_VERSION 0

#include "defines.h"
#include "log.h"
#include "result.h"
#include "array.h"
#include "hashmap.h"
#include "bounded_lru_cache.h"
#include "platform.h"
#include "threading.h"
#include "gfx.h"
#include "imgui_impl.h"
#include "buffered_stream.h"
// #include "graph.h"

#define SPECTRUM_USE_DARK_THEME
#include "imgui_spectrum.h"
#include "roboto-medium.h"

using glm::vec2;
using glm::ivec2;
using glm::vec3;
using glm::vec4;
using glm::mat4;
#include "types.h"

// TODO:
// [x] Load data from disk
// [x] First load everything from disk and decompress into chunks
// [x] Upload all chunks
// [x] Implement view choice on the CPU and update descriptors
//     [x] Likely use instanced quads or single buffer of tris (quad count is gonna be super low, so anything works here)
//     [x] Zooming and panning
// [x] Display load state in some sort of minimap in imgui
// [x] Fix delta coding (issue was delta per plane vs per whole chunk)
// [ ] Threaded loading
//      [x] Populate cache synchronously every frame
//      [x] Freelist allocator of images and descriptors (allocate worst case bound at the start)
//      [ ] Add a border and also upload below and above level
//      [ ] Async upload with fallback on not ready with copy queue
// [ ] Add framegraph?
// [ ] Reimplement with app::Application helper, compare code cut
// [ ] Mips on last chunk
// [ ] Non pow2 images and non multiple of chunk size


// The max number of used chunks is (probably it makes sense to also include neighbors here, otherwise we might trash the cache by prefetching too much).
// max_chunks_in_frame = (1 + ceil(width / chunk_width)) * (1 + ceil(height / chunk_height))
// max_chunks_in_use = max_chunks_in_frame * 3
//
// Allocator / desc management:
// - Allocate upfront descriptors and images for max_chunks_in_use -> allocs can be uninitialized
// - Every frame compute the chunks that are needed for these frame and issue requests for each of them.
// - The same chunks are kept in a ring buffer for each frame in flight, after acquiring a frame we first
//   decrement the refcount for the previous frame slot.
// - Keep a queue of chunks with 0 refcount. As an optimization we can keep the chunks marked as already
//   filled, to skip refilling them if the cache is large enough and we come back to recently visited areas.
//   this can give a tweakable "history" of cached tiles for free, if this history is as large as the image 
//   no loading will happen anymore.
// - On the CPU we keep track for each chunk of the refcount, descriptor and state (e.g. loading/loaded/empty)
//
// LRU cache:
// - Cache entries are in 3 possible states:
//   - Empty: (nothing has ever been loaded on this entry, it's free for use)
//   - In use: (some drawcall is in flight that uses this entry, this means that some chunk exists that refers to this entry and it's refcount is > 0)
//   - Not in use: (some drawcall previously referenced this entry, it still contains it's data but it is available for reuse if needed).
// - The cache is only managed by a single thread with this cycle:
//   - At start entries are initialized and vulkan images are allocated, the entries start in the Empty state
//   - Take the chunks that were used 3 frames ago and decrement the refcount of all the references.
//     If any goes to zero put this cache entry in the "Not In Use" state. This should be a Queue of entries from which we can remove from the middle.
//   - Take chunks that are needed for this frames and increment refcount of all the references.
//     If any is already in use (refcount > 0), done.
//     If one is not already in use:
//        If the chunk still points to a valid entry, remove it from the free list.
//        If the chunk is not valid grab a new one from the empty list.
//     
// [x] Full Sync:      
//     [x] issue load and return desc synchronously
//     [x] bounded LRU cache with queue
//     [x] Cache resizing when screen is resized (ideally only incrementally add)
// [ ] Threaded sync:  issue all needed loads to pool and wait for all of them to be satisfied before drawing. 
// [ ] Threaded async: do loading and uploads on a different thread, maybe can have pool for loading / decompressing and worker thread / main thread doing uploads, but wait at the end or at transfer buffer exaustion (do something smart with semaphores/counters?)
// [ ] Copy queue:     do uploads with copy queue, need to do queue transfer
// [ ] Prefetch:       after all loads are satisfied issue loads for neighbors
// [ ] Cancellation:   we can cancel neighbors from previous frame if they are not used / neighbors in this frame. (e.g. at zoom lvl 1 we prefetch lvl 0 and 2, if moving to 2 we can cancel prefetch at 0)
// [ ] Load all mode:  just increase the cache to the total size, issue all loads, done.

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

ChunkLoadContext AllocChunkLoadContext(const ZMipFile& zmip) {
    Array<u8> compressed_data(zmip.largest_compressed_chunk_size);
    Array<u8> interleaved(zmip.header.chunk_width * zmip.header.chunk_height * zmip.header.channels);
    Array<u8> deinterleaved(zmip.header.chunk_width * zmip.header.chunk_height * 4);

    ChunkLoadContext result = {
        .zmip = &zmip,
        .compressed_data = std::move(compressed_data),
        .interleaved = std::move(interleaved),
        .deinterleaved = std::move(deinterleaved),
    };
    return result;
}

usize GetChunkIndex(const ZMipFile& zmip, ChunkId c) {
    ZMipLevelInfo level = zmip.levels[c.l];
    return level.offset + (usize)c.y * level.chunks_x + c.x;
}

bool LoadChunk(ChunkLoadContext& load, ChunkId c) {
    const ZMipFile& zmip = *load.zmip;

    usize index = GetChunkIndex(zmip, c);
    usize x = c.x;
    usize y = c.y;
    usize l = c.l;
    ZMipChunk b = zmip.chunks[index];
    if (b.offset + b.size < b.offset) {
        logging::error("bigimage/parse/map", "offset + size overflow on chunk (%llu, %llu) at level %llu", x, y, l);
        return false;
    }
    if (b.offset + b.size > zmip.file.size) {
        logging::error("bigimage/parse/map", "offset + size out of bounds chunk (%llu, %llu) at level %llu", x, y, l);
        return false;
    }

    ArrayView<u8> chunk = load.compressed_data.slice(0, b.size);
    if (platform::ReadAtOffset(zmip.file, chunk, b.offset) != platform::Result::Success) {
        logging::error("bigimage/parse/chunk", "Failed to read %u bytes at offset %llu", b.size, b.offset);
        return false;
    }
    usize frame_size = ZSTD_getFrameContentSize(chunk.data, chunk.length);
    if (frame_size != load.interleaved.length) {
        logging::error("bigimage/parse/chunk", "Compressed chunk frame size %llu does not match expected size %llu", frame_size, load.interleaved.length);
        return false;
    }
    usize zstd_code = ZSTD_decompress(load.interleaved.data, load.interleaved.length, chunk.data, chunk.length);
    if (ZSTD_isError(zstd_code)) {
        logging::error("bigimage/parse/chunk", "ZSTD_decompress failed with error %s (%llu)", ZSTD_getErrorName(zstd_code), zstd_code);
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

    logging::info("bigimage/parse/header", "magic: %llu", header.magic);
    logging::info("bigimage/parse/header", "width: %llu", header.width);
    logging::info("bigimage/parse/header", "height: %llu", header.height);
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
        .chunks = std::move(chunks),
        .levels = std::move(levels),
        .largest_compressed_chunk_size = largest_compressed_chunk_size,
    };

    return zmip;
}

struct ChunkCache {
    // enum class LoadState: u32 {
    //     Unloaded = 0,
    //     Loading,
    //     Loaded,
    // };

    // Entry in the cache
    static constexpr usize INVALID_CHUNK = ~(usize)(0);
    struct Entry {
        usize chunk_index;
        u32 image_index;
    };

    // Information about a chunk, contains all possible chunks
    struct Chunk {
        // alignas(64) std::atomic<LoadState> state;
        u32 refcount;                   // Frames in flight that use this chunk
        BoundedLRUCache<Entry>::Entry* lru_entry;     // Optional entry into the LRU cache. The value in the entry indexes both the images array and descriptor set.
    };

    struct UploadWork {
        ArrayView<u8> buffer_map;
        ChunkId c;
        BlockingCounter* work_done_counter;
    };

    ChunkCache(const ZMipFile& zmip, usize cache_size, usize upload_buffers_count, usize num_workers,
               const gfx::Context& vk, const gfx::BindlessDescriptorSet bindless)
               : zmip(zmip)
               , lru(cache_size)
    {
        chunks_count = zmip.chunks.length;
        chunks = (Chunk*)AlignedAlloc(alignof(Chunk), sizeof(Chunk) * chunks_count);
        memset(chunks, 0, sizeof(Chunk) * chunks_count);

        worker_infos.resize(num_workers);
        Array<void*> worker_info_ptr(num_workers);
        for (usize i = 0; i < worker_infos.length; i++) {
            worker_infos[i] = AllocChunkLoadContext(zmip);
            worker_info_ptr[i] = &worker_infos[i];
        }
        worker_pool.init_with_worker_data(worker_info_ptr);

        sync_load_context = AllocChunkLoadContext(zmip);

        resize(cache_size, upload_buffers_count, vk, bindless);
    }

    void DestroyResources(const gfx::Context& vk) {
        for (usize i = 0; i < images.length; i++) {
            gfx::DestroyImage(&images[i], vk);
        }
        for (usize i = 0; i < upload_buffers.length; i++) {
            gfx::DestroyBuffer(&upload_buffers[i], vk);
        }
    }

    void resize(usize cache_size, usize upload_buffers_count, const gfx::Context& vk, const gfx::BindlessDescriptorSet bindless) {
        // The cache does not shrink
        if (images.length < cache_size) {
            // Add missing images
            logging::info("bigimage/cache", "Resizing cache to size %llu", cache_size);
            usize old_length = images.length;
            images.resize(cache_size);
            for (usize i = old_length; i < images.length; i++) {
                // Alloc image
                gfx::CreateImage(&images[i], vk, {
                    .width = zmip.header.chunk_width,
                    .height = zmip.header.chunk_height,
                    .format = VK_FORMAT_R8G8B8A8_UNORM,
                    .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    .memory_required_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                });

                // Write descriptor
                gfx::WriteImageDescriptor(bindless.set, vk, {
                    .view = images[i].view,
                    .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    .binding = 2,
                    .element = (u32)i,
                });

                Entry e = {
                    .chunk_index = INVALID_CHUNK,
                    .image_index = (u32)i,
                };
                BoundedLRUCache<Entry>::Entry* lru_entry = lru.alloc(std::move(e));
                lru.push(lru_entry);
            }
        }

        // The cache does not shrink
        if (upload_buffers.length < upload_buffers_count) {
            // Add missing upload buffers
            logging::info("bigimage/cache", "Adding upload buffers to size %llu", upload_buffers_count);
            usize old_length = upload_buffers.length;
            upload_buffers.resize(upload_buffers_count);

            for (usize i = old_length; i < upload_buffers.length; i++) {
                CreateBuffer(&upload_buffers[i], vk, zmip.header.chunk_width * zmip.header.chunk_height * 4, {
                    .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    .alloc_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                    .alloc_usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                });
            }

            // Add missing work items
            work_items.resize(upload_buffers_count);
        }
    }

    ~ChunkCache() {
        AlignedFree(chunks);
    }

    Chunk& get_chunk(ChunkId c) {
        usize index = GetChunkIndex(zmip, c);
        assert(index < chunks_count);
        return chunks[index];
    }

    void release_chunk(usize chunk_index) {
        // Lookup state in cache
        Chunk& chunk = chunks[chunk_index];
        assert(chunk.refcount > 0);
        assert(chunk.lru_entry);

        chunk.refcount -= 1;
        if (chunk.refcount == 0) {
            lru.push(chunk.lru_entry);
        }
    }

    static void worker_func(WorkerPool::WorkerInfo* worker_info, void* user_data) {
        UploadWork& work = *(UploadWork*)user_data;
        ChunkLoadContext& ctx = *(ChunkLoadContext*)worker_info->data;

        // Load chunk from disk
        LoadChunk(ctx, work.c);

        // Write chunk to mapped buffer
        work.buffer_map.copy_exact(ctx.deinterleaved);

        // Signal that this work item is done
        work.work_done_counter->signal();
    }

    void request_chunk_batch(ArrayView<ChunkId> chunk_ids, ArrayView<u32> output_descriptors, const gfx::Context& vk, const gfx::BindlessDescriptorSet& bindless) {
        platform::Timestamp begin = platform::GetTimestamp();

        // For now we assert that we don't have batches larger than the prepared staging buffers.
        // In theory we could tweak this and do the upload in multiple stages, which in theory
        // can also be useful to overlap loading and upload work for lower latency.
        assert(chunk_ids.length <= upload_buffers.length);

        // Arm counter to know when work is done
        work_done_counter.arm(chunk_ids.length);

        // Issue work to threads
        usize chunks_to_upload = 0;
        for (usize i = 0; i < chunk_ids.length; i++) {
            ChunkId c = chunk_ids[i];
            usize chunk_index = GetChunkIndex(zmip, c);
            assert(chunk_index < chunks_count);

            // Lookup state in cache
            Chunk& chunk = chunks[chunk_index];
            UploadWork& work = work_items[i];
            work.c = c;
            work.buffer_map = upload_buffers[i].map;
            work.work_done_counter = &work_done_counter;

            if (!chunk.lru_entry) {
                chunks_to_upload += 1;
                worker_pool.add_work({
                    .callback = worker_func,
                    .user_data = &work,
                });
            }
            else {
                // Fill chunks for which we did not submit work
                output_descriptors[i] = chunk.lru_entry->value.image_index;

                // Remove the entry from the LRU queue if we are using it again
                if (chunk.refcount == 0) {
                    lru.remove(chunk.lru_entry);
                }

                // This item generated no work, signal the counter to not count it when waiting
                work_done_counter.signal();
            }

            chunk.refcount += 1;
        }

        // Return early if no uploads needed
        if (chunks_to_upload == 0) return;

        // Wait for all work done
        work_done_counter.wait();

        platform::Timestamp mid = platform::GetTimestamp();

        // Issue upload commands to GPU
        gfx::BeginCommands(vk.sync_command_pool, vk.sync_command_buffer, vk);

        for (usize i = 0; i < chunk_ids.length; i++) {
            ChunkId c = chunk_ids[i];
            usize chunk_index = GetChunkIndex(zmip, c);
            assert(chunk_index < chunks_count);

            // Lookup state in cache
            Chunk& chunk = chunks[chunk_index];

            // Skip chunks for which we did not submit work
            if (chunk.lru_entry) {
                continue;
            }

            // Pop an entry from the LRU cache
            BoundedLRUCache<Entry>::Entry* e = lru.pop();

            // If the entry was previously used for some other chunk
            // invalidate the chunk that was holding on to it.
            if (e->value.chunk_index != INVALID_CHUNK) {
                Chunk& other = chunks[e->value.chunk_index];

                // The chunk should never be in the LRU cache if it's in use.
                assert(other.refcount == 0);

                other.lru_entry = 0;
            }

            // Fill the output descriptor
            output_descriptors[i] = e->value.image_index;

            // Store which chunk this cache entry stores now. And which entry this chunk points to.
            e->value.chunk_index = chunk_index;
            chunk.lru_entry = e;

            gfx::Image& image = images[e->value.image_index];

            // Transition image to transfer dest layout
            gfx::CmdImageBarrier(vk.sync_command_buffer, image.image, {
                .src_stage = VK_PIPELINE_STAGE_2_NONE,
                .dst_stage = VK_PIPELINE_STAGE_2_COPY_BIT,
                .src_access = 0,
                .dst_access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
                .new_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            });

            // Issue copy
            VkBufferImageCopy copy = {};
            copy.bufferOffset = 0;
            copy.bufferRowLength = zmip.header.chunk_width;
            copy.bufferImageHeight = zmip.header.chunk_height;
            copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy.imageSubresource.mipLevel = 0;
            copy.imageSubresource.baseArrayLayer = 0;
            copy.imageSubresource.layerCount = 1;
            copy.imageExtent.width = zmip.header.chunk_width;
            copy.imageExtent.height = zmip.header.chunk_height;
            copy.imageExtent.depth = 1;

            vkCmdCopyBufferToImage(vk.sync_command_buffer, upload_buffers[i].buffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

            // Transition images to shader read layout
            gfx::CmdImageBarrier(vk.sync_command_buffer, image.image, {
                .src_stage = VK_PIPELINE_STAGE_2_COPY_BIT,
                .dst_stage = VK_PIPELINE_STAGE_2_NONE,
                .src_access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .dst_access = 0,
                .old_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .new_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            });
        }

        gfx::EndCommands(vk.sync_command_buffer);

        // Submit and wait
        gfx::SubmitSync(vk);

        platform::Timestamp end = platform::GetTimestamp();
        logging::info("bigimage/cache/batch", "Chunks: %4llu / %4llu | Load: %12.6f | Upload: %12.6f", 
            chunks_to_upload, chunk_ids.length, platform::GetElapsed(begin, mid) * 1000.0, platform::GetElapsed(mid, end) * 1000.0);
    }

    u32 request_chunk_sync(ChunkId c, const gfx::Context& vk, const gfx::BindlessDescriptorSet& bindless) {
        usize chunk_index = GetChunkIndex(zmip, c);
        assert(chunk_index < chunks_count);

        // Lookup state in cache
        Chunk& chunk = chunks[chunk_index];

        // switch (chunk.state.load(std::memory_order_relaxed)) {
        //     case LoadState::Unloaded: {
                // Check if this chunk already has an entry in the cache.
                // If it has, we can just use it, otherwise we need to populate it.
        //         chunk.state.store(LoadState::Loaded, std::memory_order_relaxed);
        //     } break;
        //     case LoadState::Loading: {
        //         while (chunk.state.load(std::memory_order_relaxed) == LoadState::Loading) {}
        //         std::atomic_thread_fence(std::memory_order_acquire);
        //     } break;
        // }

        if (!chunk.lru_entry) {
            // Pop an entry from the LRU cache
            BoundedLRUCache<Entry>::Entry* e = lru.pop();

            // If the entry was previously used for some other chunk
            // invalidate the chunk that was holding on to it.
            if (e->value.chunk_index != INVALID_CHUNK) {
                Chunk& other = chunks[e->value.chunk_index];

                // The chunk should never be in the LRU cache if it's in use.
                assert(other.refcount == 0);

                other.lru_entry = 0;
            }

            // Store which chunk this cache entry stores now. And which entry this chunk points to.
            e->value.chunk_index = chunk_index;
            chunk.lru_entry = e;

            platform::Timestamp begin = platform::GetTimestamp();
            // Load from disk
            LoadChunk(sync_load_context, c);

            platform::Timestamp mid = platform::GetTimestamp();

            // Upload to GPU
            gfx::UploadImage(images[e->value.image_index], vk, sync_load_context.deinterleaved, {
                .width = zmip.header.chunk_width,
                .height = zmip.header.chunk_height,
                .format = VK_FORMAT_R8G8B8A8_UNORM,
                .current_image_layout = VK_IMAGE_LAYOUT_UNDEFINED,
                .final_image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            });

            platform::Timestamp end = platform::GetTimestamp();
            logging::info("bigimage/cache/load", "Load: %12.6f | Upload: %12.6f", platform::GetElapsed(begin, mid), platform::GetElapsed(mid, end));
        }
        else if (chunk.refcount == 0) {
            // Remove the entry from the LRU if we are using it again
            lru.remove(chunk.lru_entry);
        }

        chunk.refcount += 1;
        return chunk.lru_entry->value.image_index;
    }

    Chunk* chunks;
    usize chunks_count;
    const ZMipFile& zmip;

    Array<gfx::Image> images;
    BoundedLRUCache<Entry> lru;
    ChunkLoadContext sync_load_context;

    Array<gfx::Buffer> upload_buffers;

    ObjArray<ChunkLoadContext> worker_infos;
    Array<UploadWork> work_items;
    WorkerPool worker_pool;
    BlockingCounter work_done_counter;
};

int main(int argc, char** argv) {
    gfx::Result result;

    result = gfx::Init();
    if (result != gfx::Result::SUCCESS) {
        logging::error("bigimage", "Failed to initialize platform\n");
    }

    Array<const char*> instance_extensions = gfx::GetPresentationInstanceExtensions();
    instance_extensions.add("VK_EXT_debug_report");

    Array<const char*> device_extensions;
    device_extensions.add("VK_KHR_swapchain");
    device_extensions.add("VK_KHR_dynamic_rendering");

    gfx::Context vk = {};
    result = gfx::CreateContext(&vk, {
        .minimum_api_version = (u32)VK_API_VERSION_1_3,
        .instance_extensions = instance_extensions,
        .device_extensions = device_extensions,
        .device_features = gfx::DeviceFeatures::DYNAMIC_RENDERING | gfx::DeviceFeatures::DESCRIPTOR_INDEXING | gfx::DeviceFeatures::SYNCHRONIZATION_2,
        .enable_validation_layer = true,
        //        .enable_gpu_based_validation = true,
        });
    if (result != gfx::Result::SUCCESS) {
        logging::error("bigimage", "Failed to initialize vulkan\n");
        exit(100);
    }

    gfx::Window window = {};
    result = gfx::CreateWindowWithSwapchain(&window, vk, "XPG", 1600, 900, true);
    if (result != gfx::Result::SUCCESS) {
        logging::error("bigimage", "Failed to create vulkan window\n");
        exit(100);
    }

    gui::ImGuiImpl imgui_impl;
    gui::CreateImGuiImpl(&imgui_impl, window, vk);

    VkResult vkr;

    // Descriptors
    gfx::BindlessDescriptorSet bindless = {};
    vkr = gfx::CreateBindlessDescriptorSet(&bindless, vk, {
        .entries = {
            {
                .count = (u32)window.frames.length,
                .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            },
            {
                .count = 1,
                .type = VK_DESCRIPTOR_TYPE_SAMPLER,
            },
            {
                .count = 1024 * 1024,
                .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            },
        }
        });
    if (result != gfx::Result::SUCCESS) {
        logging::error("bigimage", "Failed to create descriptor set\n");
        exit(100);
    }

    gfx::Sampler sampler;
    vkr = gfx::CreateSampler(&sampler, vk, {
        .min_filter = VK_FILTER_NEAREST,
        .mag_filter = VK_FILTER_NEAREST,
        .mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        });
    if (result != gfx::Result::SUCCESS) {
        logging::error("bigimage", "Failed to create sampler\n");
        exit(100);
    }

    gfx::WriteSamplerDescriptor(bindless.set, vk, {
        .sampler = sampler.sampler,
        .binding = 1,
        .element = 0,
        });

    // Pipeline
    Array<u8> vert_code;
    if (platform::ReadEntireFile("res/bigimage.vert.spirv", &vert_code) != platform::Result::Success) {
        logging::error("bigimage", "Failed to read vertex shader\n");
        exit(100);
    }
    gfx::Shader vert_shader = {};
    vkr = gfx::CreateShader(&vert_shader, vk, vert_code);
    if (result != gfx::Result::SUCCESS) {
        logging::error("bigimage", "Failed to create vertex shader\n");
        exit(100);
    }

    Array<u8> frag_code;
    if (platform::ReadEntireFile("res/bigimage.frag.spirv", &frag_code) != platform::Result::Success) {
        logging::error("bigimage", "Failed to read fragment shader\n");
        exit(100);
    }
    gfx::Shader frag_shader = {};
    vkr = gfx::CreateShader(&frag_shader, vk, frag_code);
    if (result != gfx::Result::SUCCESS) {
        logging::error("bigimage", "Failed to create fragment shader\n");
        exit(100);
    }

    gfx::GraphicsPipeline pipeline = {};
    vkr = gfx::CreateGraphicsPipeline(&pipeline, vk, {
        .stages = {
            {
                .shader = vert_shader,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
            },
            {
                .shader = frag_shader,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
        },
        .vertex_bindings = {
            {
                .binding = 0,
                .stride = sizeof(glm::vec2),
            },
        },
        .vertex_attributes = {
            {
                .location = 0,
                .binding = 0,
                .format = VK_FORMAT_R32G32_SFLOAT,
            },
        },
        .push_constants = {
            {
                .offset = 0,
                .size = 32,
            },
        },
        .descriptor_sets = {
            bindless.layout,
        },
        .attachments = {
            {
                .format = window.swapchain_format,
            },
        },
    });

    // Vertex data
    struct Vertex {
        vec2 pos;
    };

    ArrayFixed<Vertex, 6> vertices(6);
    vertices[0] = { vec2(0.0f, 0.0f) };
    vertices[1] = { vec2(1.0f, 0.0f) };
    vertices[2] = { vec2(1.0f, 1.0f) };
    vertices[3] = { vec2(1.0f, 1.0f) };
    vertices[4] = { vec2(0.0f, 1.0f) };
    vertices[5] = { vec2(0.0f, 0.0f) };
    size_t V = vertices.length;

    gfx::Buffer vertex_buffer = {};
    vkr = gfx::CreateBufferFromData(&vertex_buffer, vk, vertices.as_bytes(), {
            .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            .alloc_required_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            .alloc_preferred_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        });
    assert(vkr == VK_SUCCESS);

    //ZMipFile zmip = LoadZmipFile("N:\\scenes\\hubble\\heic0601a_test_delta.zmip");
    ZMipFile zmip = LoadZmipFile("N:\\scenes\\hubble\\heic0601a_256.zmip");

    // Read image
    struct GpuChunk {
        vec2 position;
        u32 desc_index;
        u32 _padding;
    };

    struct App {
        // - Window
        bool wait_for_events = false;
        bool closed = false;

        // - Application data
        platform::Timestamp last_frame_timestamp;
        u64 current_frame = 0;  // Total frame index, always increasing
        vec2 offset = vec2(0, 0);
        s32 zoom = 0;
        s32 max_zoom = 0;
        bool first_frame_done = false;
        ivec2 drag_start_offset = ivec2(0, 0);
        ivec2 drag_start = ivec2(0, 0);
        bool dragging = false;
        bool show_grid = false;
        bool batched_chunk_upload = true;
        // bool batched_chunk_upload = false;

        Array<GpuChunk> gpu_chunks;
        Array<usize> cpu_chunks_data;
        Array<ArrayView<usize>> cpu_chunks_per_frame;
        Array<ChunkId> batch_inputs;
        Array<u32> batch_outputs;

        usize total_max_chunks = 0;

        // - Rendering
        VkPipeline pipeline;
        VkPipelineLayout layout;
        VkDescriptorSet descriptor_set;
        Array<gfx::Buffer> chunks_buffers; // Buffer containing chunk metadata, one per frame in flight
        gfx::Buffer vertex_buffer;
        s32 descriptor_count;
        u32 frame_index = 0; // Rendering frame index, wraps around at the number of frames in flight
    };

    // USER: application
    App app = {};
    app.last_frame_timestamp = platform::GetTimestamp();
    app.pipeline = pipeline.pipeline;
    app.layout = pipeline.layout;
    app.descriptor_set = bindless.set;
    app.descriptor_count = (s32)zmip.chunks.length;
    app.chunks_buffers = Array<gfx::Buffer>(window.frames.length);
    app.gpu_chunks = Array<GpuChunk>();
    app.cpu_chunks_data = Array<usize>();
    app.cpu_chunks_per_frame = Array<ArrayView<usize>>(window.frames.length);
    app.vertex_buffer = vertex_buffer;
    app.max_zoom = (s32)(zmip.levels.length - 1);

    ChunkCache cache(zmip, 0, 0, 1, vk, bindless);

    auto MouseMoveEvent = [&app](ivec2 pos) {
        if (app.dragging) {
            ivec2 delta = pos - app.drag_start;
            app.offset = delta + app.drag_start_offset;
        }
    };

    auto MouseButtonEvent = [&app](ivec2 pos, gfx::MouseButton button, gfx::Action action, gfx::Modifiers mods) {
        if (ImGui::GetIO().WantCaptureMouse) return;

        if (button == gfx::MouseButton::Left) {
            if (action == gfx::Action::Press && !app.dragging) {
                app.dragging = true;
                app.drag_start = pos;
                app.drag_start_offset = app.offset;
            }
            else if (action == gfx::Action::Release && app.dragging) {
                app.dragging = false;
                app.drag_start = ivec2(0, 0);
                app.drag_start_offset = ivec2(0, 0);
            }
        }
    };

    auto MouseScrollEvent = [&app](ivec2 pos, ivec2 scroll) {
        if (ImGui::GetIO().WantCaptureMouse) return;

        ivec2 old_image_pos = (pos - (ivec2)app.offset) << app.zoom;
        app.zoom = Clamp(app.zoom - scroll.y, 0, (s32)app.max_zoom);
        ivec2 new_image_pos = (pos - (ivec2)app.offset) << app.zoom;
        app.offset += (new_image_pos - old_image_pos) >> app.zoom;
    };

    auto Draw = [&app, &vk, &window, &bindless, &zmip, &cache]() {
        if (app.closed) return;

        platform::Timestamp timestamp = platform::GetTimestamp();
        float dt = (float)platform::GetElapsed(app.last_frame_timestamp, timestamp);
        app.last_frame_timestamp = timestamp;

        gfx::SwapchainStatus swapchain_status = UpdateSwapchain(&window, vk);
        if (swapchain_status == gfx::SwapchainStatus::FAILED) {
            logging::error("bigimg/draw", "Swapchain update failed\n");
            exit(101);
        }
        else if (swapchain_status == gfx::SwapchainStatus::MINIMIZED) {
            return;
        }

        if (swapchain_status == gfx::SwapchainStatus::RESIZED || !app.first_frame_done) {
            app.first_frame_done = true;

            // USER: resize (e.g. framebuffer sized elements)
            ivec2 view_size = ivec2(window.fb_width, window.fb_height);
            ivec2 chunk_size = ivec2(zmip.header.chunk_width, zmip.header.chunk_height);
            ivec2 max_chunks = (view_size + chunk_size - ivec2(1, 1)) / chunk_size + ivec2(1, 1);
            usize total_max_chunks = (u64)max_chunks.x * (u64)max_chunks.y;
            if (total_max_chunks > app.total_max_chunks) {
                app.total_max_chunks = total_max_chunks;

                for (usize i = 0; i < app.chunks_buffers.length; i++) {
                    gfx::DestroyBuffer(&app.chunks_buffers[i], vk);
                    gfx::CreateBuffer(&app.chunks_buffers[i], vk, app.total_max_chunks * sizeof(GpuChunk), {
                        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        .alloc_required_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                        .alloc_preferred_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    });
                    gfx::WriteBufferDescriptor(bindless.set, vk, {
                        .buffer = app.chunks_buffers[i].buffer,
                        .binding = 0,
                        .element = (u32)i,
                    });
                }

                // Resize buffer for storing gpu data to transfer to the gpu each frame
                app.gpu_chunks.resize(app.total_max_chunks);

                // Release all the chunks in use on resize
                for (usize i = 0; i < app.cpu_chunks_per_frame.length; i++) {
                    for (usize j = 0; j < app.cpu_chunks_per_frame[i].length; j++) {
                        cache.release_chunk(app.cpu_chunks_per_frame[i][j]);
                    }
                }

                // Resize buffers for chunks in flight
                app.cpu_chunks_data.resize(app.total_max_chunks * app.cpu_chunks_per_frame.length);
                for (usize i = 0; i < app.cpu_chunks_per_frame.length; i++) {
                    app.cpu_chunks_per_frame[i] = app.cpu_chunks_data.slice(i * app.total_max_chunks, 0);
                }

                // Resize cache
                cache.resize(total_max_chunks * window.frames.length, total_max_chunks, vk, bindless);
                app.batch_inputs.resize(total_max_chunks);
                app.batch_outputs.resize(total_max_chunks);
            }
        }

        // Acquire current frame
        gfx::Frame& frame = gfx::WaitForFrame(&window, vk);
        gfx::Result ok = gfx::AcquireImage(&frame, &window, vk);
        if (ok != gfx::Result::SUCCESS) {
            return;
        }

        // Chunks
        ivec2 offset = app.offset;                                                    // In screen pixel
        ivec2 view_size = ivec2(window.fb_width, window.fb_height);                   // In screen pixel
        ivec2 img_size = ivec2(zmip.header.width, zmip.header.height);                // In image pixels
        ivec2 chunk_size = ivec2(zmip.header.chunk_width, zmip.header.chunk_height);  // In image pixels

        // Zoom dependant
        ivec2 z_chunk_size = chunk_size << app.zoom;                                  // In image pixels
        ivec2 chunks = (img_size + z_chunk_size - ivec2(1, 1)) / z_chunk_size;        // Total chunks in image at this zoom level
        ivec2 remainder = (chunk_size - (view_size - offset) % chunk_size);

        // Chunks in flight
        ArrayView<usize>& cpu_chunks = app.cpu_chunks_per_frame[app.frame_index];

        // Decrease refcount of old frame chunks
        for (usize i = 0; i < cpu_chunks.length; i++) {
            cache.release_chunk(cpu_chunks[i]);
        }

        // Reset chunks of the current frame.
        cpu_chunks.length = 0;
        app.gpu_chunks.length = 0;
        app.batch_inputs.length = 0;

        for (s32 y = -offset.y; y < view_size.y - offset.y + remainder.y; y += chunk_size.y) {
            for (s32 x = -offset.x; x < view_size.x - offset.x + remainder.x; x += chunk_size.x) {
                ivec2 image_coords = ivec2(x, y) << app.zoom;
                ivec2 chunk = image_coords / z_chunk_size;

                // Skip chunks that are of bounds of the image
                if (!((chunk.x >= 0 && chunk.x < chunks.x) && (chunk.y >= 0 && chunk.y < chunks.y))) continue;

                ChunkId id((u32)chunk.x, (u32)chunk.y, (u32)app.zoom);

                u32 desc_index = UINT32_MAX;
                if (app.batched_chunk_upload) {
                    app.batch_inputs.add(id);
                }
                else {
                    desc_index = cache.request_chunk_sync(id, vk, bindless);
                }
                GpuChunk c = {
                    .position = offset + chunk * chunk_size,
                    .desc_index = desc_index,
                };

                // Check that our upper bound of max chunks is actually respected
                assert(cpu_chunks.length < app.total_max_chunks);
                assert(app.gpu_chunks.length < app.total_max_chunks);

                app.gpu_chunks.add(c);
                cpu_chunks[cpu_chunks.length++] = GetChunkIndex(zmip, id);
            }
        }

        // Upload 
        if (app.batched_chunk_upload) {
            app.batch_outputs.resize(app.batch_inputs.length);
            cache.request_chunk_batch(app.batch_inputs, app.batch_outputs, vk, bindless);
            for (usize i = 0; i < app.batch_outputs.length; i++) {
                app.gpu_chunks[i].desc_index = app.batch_outputs[i];
            }
        }

        {
            // Upload chunks
            gfx::Buffer& buffer = app.chunks_buffers[app.frame_index];

            void* addr = 0;
            VkResult vkr = vmaMapMemory(vk.vma, buffer.allocation, &addr);
            if (vkr != VK_SUCCESS) {
                logging::error("bigimage/draw", "Failed to map chunk buffer memory");
                exit(102);
            }

            ArrayView<u8> map((u8*)addr, app.gpu_chunks.size_in_bytes());
            map.copy_exact(app.gpu_chunks.as_bytes());

            vmaUnmapMemory(vk.vma, buffer.allocation);
        }



        {
            gui::BeginFrame();

            // USER: gui
            gui::DrawStats(dt, window.fb_width, window.fb_height);

            //ImGui::ShowDemoWindow();
            int32_t texture_index = 0;
            if (ImGui::Begin("Editor")) {
                ImGui::InputInt("Zoom", &app.zoom);
                app.zoom = Clamp(app.zoom, 0, (s32)zmip.levels.length - 1);

                ImGui::DragFloat2("Offset", &app.offset.x);

                ImGui::Checkbox("Show grid", &app.show_grid);

                ImGui::Separator();

                // Draw minimap
                ImDrawList* draw_list = ImGui::GetWindowDrawList();
                ImVec2 corner = ImGui::GetCursorScreenPos();
                f32 size = 5.0f;
                f32 stride = 7.0f;
                ivec2 img_size = ivec2(zmip.header.width, zmip.header.height);                // In image pixels
                for (s32 level = 0; level < (s32)zmip.levels.length; level++) {
                    ivec2 chunk_size = ivec2(zmip.header.chunk_width, zmip.header.chunk_height) << level;  // In image pixels
                    ivec2 chunks = (img_size + chunk_size - ivec2(1, 1)) / chunk_size;  // Total chunks in image at this zoom level
                    for (s32 y = 0; y < chunks.y; y++) {
                        for (s32 x = 0; x < chunks.x; x++) {
                            ChunkCache::Chunk& c = cache.get_chunk(ChunkId(x, y, level));
                            u32 color = 0xFF0000FF;
                            // ChunkCache::LoadState state = c.state.load(std::memory_order_relaxed);
                            if (c.lru_entry) {
                                if (c.refcount > 0) {
                                    color = 0xFF00FF00;
                                }
                                else {
                                    color = 0xFF00FFFF;
                                }
                            }
                            draw_list->AddRectFilled(corner + ImVec2(x * stride, y * stride), corner + ImVec2(x * stride + size, y * stride + size), color);
                        }
                    }

                    if (level & 1) {
                        corner.x += chunks.x * stride + 5.0f;
                    }
                    else {
                        corner.y += chunks.y * stride + 5.0f;
                    }
                }
            }
            ImGui::End();

            gui::EndFrame();
        }

        {
            gfx::BeginCommands(frame.command_pool, frame.command_buffer, vk);

            // USER: draw commands
            gfx::CmdImageBarrier(frame.command_buffer, frame.current_image, {
                .src_stage = VK_PIPELINE_STAGE_2_NONE,
                .dst_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .src_access = 0,
                .dst_access = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
                .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                });

            //VkClearColorValue color = { 0.1f, 0.2f, 0.4f, 1.0f };
            VkClearColorValue color = { 0.1f, 0.1f, 0.1f, 1.0f };
            VkRenderingAttachmentInfo attachment_info = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
            attachment_info.imageView = frame.current_image_view;
            attachment_info.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            attachment_info.resolveMode = VK_RESOLVE_MODE_NONE;
            attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attachment_info.clearValue.color = color;

            VkRenderingInfo rendering_info = { VK_STRUCTURE_TYPE_RENDERING_INFO };
            rendering_info.renderArea.extent.width = window.fb_width;
            rendering_info.renderArea.extent.height = window.fb_height;
            rendering_info.layerCount = 1;
            rendering_info.viewMask = 0;
            rendering_info.colorAttachmentCount = 1;
            rendering_info.pColorAttachments = &attachment_info;
            rendering_info.pDepthAttachment = 0;
            vkCmdBeginRenderingKHR(frame.command_buffer, &rendering_info);

            vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app.pipeline);

            VkDeviceSize offsets[1] = { 0 };
            vkCmdBindVertexBuffers(frame.command_buffer, 0, 1, &app.vertex_buffer.buffer, offsets);

            VkViewport viewport = {};
            viewport.width = (f32)window.fb_width;
            viewport.height = (f32)window.fb_height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(frame.command_buffer, 0, 1, &viewport);

            VkRect2D scissor = {};
            scissor.extent.width = window.fb_width;
            scissor.extent.height = window.fb_height;
            vkCmdSetScissor(frame.command_buffer, 0, 1, &scissor);

            vkCmdBindDescriptorSets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app.layout, 0, 1, &app.descriptor_set, 0, 0);

            struct Constants {
                vec2 scale;
                vec2 inv_window_size;
                vec2 inv_scale;
                u32 frame_id;
                u32 flags;
            };

            Constants constants = {
                .scale = vec2(chunk_size),
                .inv_window_size = vec2(1.0f / (f32)window.fb_width, 1.0f / (f32)window.fb_height),
                .inv_scale = 1.0f / vec2(chunk_size),
                .frame_id = (u32)app.frame_index,
                .flags = (u32)app.show_grid << 0,
            };
            vkCmdPushConstants(frame.command_buffer, app.layout, VK_SHADER_STAGE_ALL, 0, sizeof(Constants), &constants);

            vkCmdDraw(frame.command_buffer, 6, (u32)app.gpu_chunks.length, 0, 0);

            // Draw GUI

            ImDrawData* draw_data = ImGui::GetDrawData();
            ImGui_ImplVulkan_RenderDrawData(draw_data, frame.command_buffer);

            vkCmdEndRenderingKHR(frame.command_buffer);

            gfx::CmdImageBarrier(frame.command_buffer, frame.current_image, {
                .src_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .dst_stage = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                .src_access = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                .dst_access = 0,
                .old_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .new_layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                });

            gfx::EndCommands(frame.command_buffer);
        }

        VkResult vkr;
        vkr = gfx::Submit(frame, vk, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        assert(vkr == VK_SUCCESS);

        vkr = gfx::PresentFrame(&window, &frame, vk);
        assert(vkr == VK_SUCCESS);

        app.current_frame += 1;
        app.frame_index = (app.frame_index + 1) % (u32)window.frames.length;
    };

    ImGui_ImplGlfw_RestoreCallbacks(window.window);

    gfx::SetWindowCallbacks(&window, {
        .mouse_move_event = MouseMoveEvent,
        .mouse_button_event = MouseButtonEvent,
        .mouse_scroll_event = MouseScrollEvent,
        .draw = Draw,
    });

    ImGui_ImplGlfw_InstallCallbacks(window.window);

    while (true) {
        gfx::ProcessEvents(app.wait_for_events);

        if (gfx::ShouldClose(window)) {
            logging::info("bigimg", "Window closed");
            app.closed = true;
            break;
        }

        // Draw
        Draw();
    };

    // Wait
    gfx::WaitIdle(vk);

    // USER: cleanup
    // for (usize i = 0; i < all_images.length; i++) {
    //     gfx::DestroyImage(&all_images[i], vk);
    // }
    cache.DestroyResources(vk);

    for (usize i = 0; i < app.chunks_buffers.length; i++) {
        gfx::DestroyBuffer(&app.chunks_buffers[i], vk);
    }
    gfx::DestroyBuffer(&vertex_buffer, vk);

    gfx::DestroySampler(&sampler, vk);
    gfx::DestroyShader(&vert_shader, vk);
    gfx::DestroyShader(&frag_shader, vk);
    gfx::DestroyGraphicsPipeline(&pipeline, vk);
    gfx::DestroyBindlessDescriptorSet(&bindless, vk);

    // Gui
    gui::DestroyImGuiImpl(&imgui_impl, vk);

    // Window
    gfx::DestroyWindowWithSwapchain(&window, vk);

    // Context
    gfx::DestroyContext(&vk);
}
