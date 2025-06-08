#include "chunk_cache.h"

#include <xpg/log.h>
#include <xpg/platform.h>

using namespace xpg;

ChunkCache::ChunkCache(const ZMipFile& zmip, usize cache_size, usize upload_buffers_count, usize num_workers, usize num_frames_in_flight,
               const gfx::Context& vk, const gfx::DescriptorSet& descriptor_set)
               : zmip(zmip)
               , lru(cache_size)
               , upload_buffers(num_frames_in_flight)
{
    // Allocate space for chunk metadata
    chunks.resize(zmip.chunks.length);

    // Initialize workers for async
    worker_infos.resize(num_workers);
    Array<void*> worker_info_ptr(num_workers);
    for (usize i = 0; i < worker_infos.length; i++) {
        worker_infos[i] = AllocChunkLoadContext(zmip);
        worker_info_ptr[i] = &worker_infos[i];
    }
    worker_pool.init_with_worker_data(worker_info_ptr);

    // Initialize laod context for sync operation
    sync_load_context = AllocChunkLoadContext(zmip);

    // Resize the cache with the starting size
    resize(cache_size, upload_buffers_count, vk, descriptor_set);
}

void ChunkCache::destroy_resources(const gfx::Context& vk) {
    for (usize i = 0; i < images.length; i++) {
        gfx::DestroyImage(&images[i], vk);
    }
    for (usize frame_index = 0; frame_index < upload_buffers.length; frame_index++) {
        for (usize i = 0; i < upload_buffers[frame_index].length; i++) {
            gfx::DestroyBuffer(&upload_buffers[frame_index][i], vk);
        }
    }
}

void ChunkCache::resize(usize cache_size, usize upload_buffers_count, const gfx::Context& vk, const gfx::DescriptorSet descriptor_set) {
    // The cache does not shrink
    if (images.length < cache_size) {
        // Add missing images
        logging::info("bigimage/cache", "Resizing cache to size %zu", cache_size);
        usize old_length = images.length;
        images.resize(cache_size);
        for (usize i = old_length; i < images.length; i++) {
            // Alloc image
            gfx::CreateImage(&images[i], vk, {
                .width = zmip.header.chunk_width,
                .height = zmip.header.chunk_height,
                .format = VK_FORMAT_R8G8B8A8_UNORM,
                .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                .alloc = gfx::AllocPresets::Device,
            });

            // Write descriptor
            gfx::WriteImageDescriptor(descriptor_set.set, vk, {
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
            PoolQueue<Entry>::Entry* lru_entry = lru.alloc(move(e));
            lru.push(lru_entry);
        }
        if(images.length > 0) {
            VmaAllocationInfo info;
            vmaGetAllocationInfo(vk.vma, images[0].allocation, &info);
            chunk_memory_size = info.size;
        } else {
            chunk_memory_size = 0;
        }
    }

    // The cache does not shrink
    if (upload_buffers.length < upload_buffers_count) {
        // Add missing upload buffers
        logging::info("bigimage/cache", "Adding upload buffers to size %zu", upload_buffers_count);
        for(usize frame_index = 0; frame_index < upload_buffers.length; frame_index++) {
            usize old_length = upload_buffers[frame_index].length;
            upload_buffers[frame_index].resize(upload_buffers_count);

            for (usize i = old_length; i < upload_buffers[frame_index].length; i++) {
                CreateBuffer(&upload_buffers[frame_index][i], vk, zmip.header.chunk_width * zmip.header.chunk_height * 4, {
                    .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    .alloc = gfx::AllocPresets::HostWriteCombining,
                });
            }
        }

        // Add missing work items
        work_items.resize(upload_buffers_count);
    }
}

void ChunkCache::release_chunk(usize chunk_index) {
    // Lookup state in cache
    Chunk& chunk = chunks[chunk_index];
    assert(chunk.refcount > 0);
    assert(chunk.lru_entry);

    chunk.refcount -= 1;
    if (chunk.refcount == 0) {
        lru.push(chunk.lru_entry);
    }
}

void ChunkCache::worker_func(WorkerPool::WorkerInfo* worker_info, void* user_data) {
    Work& work = *(Work*)user_data;
    ChunkLoadContext& ctx = *(ChunkLoadContext*)worker_info->data;

    // Load chunk from disk
    LoadChunk(ctx, work.c);

    // Write chunk to mapped buffer
    work.buffer_map.copy_exact(ctx.deinterleaved);

    // Signal that this work item is done
    work.work_done_counter->signal();
}

void ChunkCache::request_chunk_batch(ArrayView<ChunkId> chunk_ids, ArrayView<u32> output_descriptors, const gfx::Context& vk, const gfx::DescriptorSet& descriptor_set, VkCommandBuffer cmd, u32 frame_index) {
    // Debug check that the same chunk is not requested twice
    for (usize i = 0; i < chunk_ids.length; i++) {
        for (usize j = i + 1; j < chunk_ids.length; j++) {
            assert(GetChunkIndex(zmip, chunk_ids[i]) != GetChunkIndex(zmip, chunk_ids[j]));
        }
    }

    // For now we assert that we don't have batches larger than the prepared staging buffers.
    // In theory we could tweak this and do the upload in multiple stages, which
    // can also be useful to overlap loading and upload work for lower latency.
    // This would require multiple submits and waits, which may not be optimal
    // to do on the frame queue, ideally this could be overlapped with other work
    // and potentially even with previous frame (allowing lag in the presentation).
    //
    // For a GUI application like this I think having slightly slower frames but
    // with correct contents makes more sense, scrolling seems to be smooth enough
    // with delays and I'd rather not have to worry about knowing if the frame i'm
    // looking at finished loading or not.
    assert(chunk_ids.length <= upload_buffers[frame_index].length);

    // If no work to upload, early out
    if (chunk_ids.length == 0) {
        return;
    }

    // Arm counter to know when work is done
    work_done_counter.arm(chunk_ids.length);

    // Issue work to threads
    usize uploaded_chunks = 0;
    for (usize i = 0; i < chunk_ids.length; i++) {
        ChunkId c = chunk_ids[i];
        usize chunk_index = GetChunkIndex(zmip, c);

        // Lookup state in cache
        Chunk& chunk = chunks[chunk_index];
        Work& work = work_items[i];
        work.c = c;
        work.buffer_map = upload_buffers[frame_index][i].map;
        work.work_done_counter = &work_done_counter;

        if (!chunk.lru_entry) {
            uploaded_chunks += 1;
            worker_pool.add_work({
                .callback = worker_func,
                .user_data = &work,
            });
        }
        else {
            output_descriptors[i] = chunk.lru_entry->value.image_index;

            if (chunk.refcount == 0) {
                // Remove the entry from the LRU queue if we are using it again
                lru.remove(chunk.lru_entry);
            }

            // This item generated no work, signal the counter to not count it when waiting
            work_done_counter.signal();
        }

        chunk.refcount += 1;
    }

    // Wait for all work done
    platform::Timestamp begin = platform::GetTimestamp();
    work_done_counter.wait();
    assert(work_done_counter.count == 0);
    platform::Timestamp mid = platform::GetTimestamp();

    if (uploaded_chunks == 0) {
        return;
    }

    // Issue upload commands to GPU
    for (usize i = 0; i < chunk_ids.length; i++) {
        ChunkId c = chunk_ids[i];
        usize chunk_index = GetChunkIndex(zmip, c);

        // Lookup state in cache
        Chunk& chunk = chunks[chunk_index];

        // Fill and skip chunks for which we did not submit work
        if (chunk.lru_entry) {
            continue;
        }

        // Pop an entry from the LRU cache
        PoolQueue<Entry>::Entry* e = lru.pop();

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
        gfx::CmdImageBarrier(cmd, {
            .src_stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .src_access = VK_ACCESS_2_SHADER_READ_BIT,
            .dst_stage = VK_PIPELINE_STAGE_2_COPY_BIT,
            .dst_access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            .new_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .image = image.image,
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

        vkCmdCopyBufferToImage(cmd, upload_buffers[frame_index][i].buffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

        // Transition images to shader read layout
        gfx::CmdImageBarrier(cmd, {
            .src_stage = VK_PIPELINE_STAGE_2_COPY_BIT,
            .src_access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dst_stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .dst_access = VK_ACCESS_2_SHADER_READ_BIT,
            .old_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .new_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .image = image.image,
        });
    }
    platform::Timestamp end = platform::GetTimestamp();

    // logging::info("bigimage/cache/batch", "Chunks: %4llu / %4llu | Load: %12.6f | Commands: %12.6f",  uploaded_chunks, chunk_ids.length, platform::GetElapsed(begin, mid) * 1000.0, platform::GetElapsed(end, end) * 1000.0);
}

u32 ChunkCache::request_chunk_sync(ChunkId c, const gfx::Context& vk, const gfx::DescriptorSet& descriptor_set) {
    usize chunk_index = GetChunkIndex(zmip, c);

    // Lookup state in cache
    Chunk& chunk = chunks[chunk_index];

    if (!chunk.lru_entry) {
        // Pop an entry from the LRU cache
        PoolQueue<Entry>::Entry* e = lru.pop();

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
