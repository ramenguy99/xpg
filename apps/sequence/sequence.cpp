#define DIRECT_UPLOAD 0
#define DEVICE_LOCAL_VERTICES 1
#define SYNC_QUEUE 0

#include <xpg/gui.h>
#include <xpg/buffered_stream.h>
#include <xpg/log.h>
#include <xpg/platform.h>

#include <math.h>

using glm::vec2;
using glm::ivec2;
using glm::vec3;
using glm::vec4;
using glm::mat4;

using namespace xpg;

#include "types.h"

int main(int argc, char** argv) {
    platform::File file = {};
    platform::Result open_result = OpenFile(argv[1], &file);
    assert(open_result == platform::Result::Success);

    Array<u8> header(12);
    ReadAtOffset(file, header, 0);

    ArrayView<u8> parser = header;
    usize N = parser.consume<u32>();
    usize V = parser.consume<u32>();
    usize I = parser.consume<u32>();

    Array<u32> indices(I);
    ReadAtOffset(file, indices.as_bytes(), N * V * sizeof(vec3) + header.length);

    gfx::Result result;
    result = gfx::Init();
    if (result != gfx::Result::SUCCESS) {
        logging::error("sequence", "Failed to initialize platform\n");
    }

    gfx::Context vk = {};
    result = gfx::CreateContext(&vk, {
        .minimum_api_version = (u32)VK_API_VERSION_1_1,
        .required_features = gfx::DeviceFeatures::DYNAMIC_RENDERING | gfx::DeviceFeatures::DESCRIPTOR_INDEXING | gfx::DeviceFeatures::SYNCHRONIZATION_2,
        .preferred_frames_in_flight = 2,
        .enable_validation_layer = true,
        // .enable_gpu_based_validation = false,
    });

#if SYNC_QUEUE
    vk.copy_queue = VK_NULL_HANDLE;
#endif

    if (result != gfx::Result::SUCCESS) {
        logging::error("sequence", "Failed to initialize vulkan\n");
        exit(100);
    }
    gfx::Window window = {};
    if (gfx::CreateWindowWithSwapchain(&window, vk, "XPG", 1600, 900) != gfx::Result::SUCCESS) {
        printf("Failed to create vulkan window\n");
        return 1;
    }

    struct App {
        // Swapchain frames, index wraps around at the number of frames in flight.
        u32 frame_index;
        // Total frame index.
        u64 current_frame;

        bool force_swapchain_update;
        bool wait_for_events;
        bool closed;

        // Application data.
        platform::Timestamp last_frame_timestamp;
        ArrayFixed<f32, 64> frame_times;
        Array<gfx::DescriptorSet> descriptor_sets;
        Array<gfx::Buffer> uniform_buffers;
        gfx::Image depth_buffer;

        // Playback
        bool playback_enabled;
        u32 num_playback_frames;
        u32 playback_frame;
        f64 playback_delta;

        BufferedStream<platform::FileReadWork> mesh_stream;
        platform::File mesh_file;
        u32 num_vertices;
        u32 num_indices;

        // Rendering
        Array<gfx::Buffer> vertex_buffers_upload;
        Array<gfx::Buffer> vertex_buffers_gpu;
        gfx::Buffer index_buffer;
        VkPipeline pipeline;
        VkPipelineLayout layout;
        Array<VkSemaphore> copy_done_semaphores;
        Array<VkSemaphore> render_done_semaphores;
    };

    VkResult vkr;


    u64 size = V * sizeof(vec3);

    logging::info("sequence", "V %zu | I %zu | S %zu", V, I, size);

    u32 WORKERS = 4;
    u32 BUFFER_SIZE = (u32)8;

    Array<gfx::Buffer> vertex_buffers_upload(BUFFER_SIZE);
#if !DIRECT_UPLOAD
    for(usize i = 0; i < BUFFER_SIZE; i++) {
        vkr = gfx::CreateBuffer(&vertex_buffers_upload[i], vk, sizeof(vec3) * V, {
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .alloc = gfx::AllocPresets::Host,
        });
        assert(vkr == VK_SUCCESS);
        VkMemoryPropertyFlags upload_properties = {};
        vmaGetAllocationMemoryProperties(vk.vma, vertex_buffers_upload[i].allocation, &upload_properties);
    }
#endif

    Array<gfx::Buffer> vertex_buffers_gpu(window.frames.length);
    for(usize i = 0; i < window.frames.length; i++) {
        vkr = gfx::CreateBuffer(&vertex_buffers_gpu[i], vk, sizeof(vec3) * V, {
            .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    #if !DIRECT_UPLOAD
                | VK_BUFFER_USAGE_TRANSFER_DST_BIT
    #endif
            ,

    #if DIRECT_UPLOAD
            .alloc = gfx::AllocPresets::DeviceMapped,
    #else
    #if DEVICE_LOCAL_VERTICES
            .alloc = gfx::AllocPresets::Device,
    #else
            .alloc = gfx::AllocPresets::Host,
    #endif
    #endif
        });
        assert(vkr == VK_SUCCESS);
        VkMemoryPropertyFlags gpu_properties = {};
        vmaGetAllocationMemoryProperties(vk.vma, vertex_buffers_gpu[i].allocation, &gpu_properties);
    }

#if DIRECT_UPLOAD
    Array<u8, 4096> loaded_data(AlignUp(size, 4096) * BUFFER_SIZE);
    ArrayView<u8> loaded_data_view = loaded_data;
#else
    ArrayView<gfx::Buffer> vertex_buffers_upload_view = vertex_buffers_upload;
#endif

    WorkerPool pool;
    pool.init(WORKERS);

    using platform::FileReadWork;
    BufferedStream<FileReadWork> mesh_stream(N, BUFFER_SIZE, &pool,
        // Init
        [=](u64 index, u64 buffer_index, bool high_priority) mutable {
            FileReadWork w = {};

            w.file = file;
            w.offset = 12 + size * index;
#if DIRECT_UPLOAD
            w.buffer = loaded_data_view.slice(buffer_index * AlignUp(size, 4096), size);
#else
            w.buffer = vertex_buffers_upload_view[buffer_index].map;
#endif
            w.do_chunks = true;

            return w;
    },
        // Fill
        [](FileReadWork* w) {
            u64 bytes_left = w->buffer.length - w->bytes_read;

            u64 chunk_size = w->do_chunks ? Min((u64)128 * 1024, bytes_left) : bytes_left;
            ReadAtOffset(w->file, w->buffer.slice(w->bytes_read, chunk_size), w->offset + w->bytes_read);
            w->bytes_read += chunk_size;

            //printf("Read data: %llu  %llu\n", w->buffer.length, w->offset);
            return w->bytes_read == w->buffer.length;
        }
    );


    gfx::Buffer index_buffer = {};
    vkr = gfx::CreateBufferFromData(&index_buffer, vk, indices.as_bytes(), {
        .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        .alloc = gfx::AllocPresets::DeviceMapped,
    });
    assert(vkr == VK_SUCCESS);

    // Create graphics pipeline.
    Array<u8> vertex_code;
    if (platform::ReadEntireFile("res/basic.vert.spirv", &vertex_code) != platform::Result::Success) {
        logging::error("sequence", "Failed to read vertex shader");
        exit(100);
    }
    Array<u8> fragment_code;
    if (platform::ReadEntireFile("res/basic.frag.spirv", &fragment_code) != platform::Result::Success) {
        logging::error("sequence", "Failed to read fragment shader");
        exit(100);
    }

    gfx::Shader vertex_shader = {};
    vkr = gfx::CreateShader(&vertex_shader, vk, vertex_code);
    if (result != gfx::Result::SUCCESS) {
        logging::error("sequence", "Failed to create vertex shader");
        exit(100);
    }

    gfx::Shader fragment_shader = {};
    vkr = gfx::CreateShader(&fragment_shader, vk, fragment_code);
    if (result != gfx::Result::SUCCESS) {
        logging::error("sequence", "Failed to create fragment shader");
        exit(100);
    }

    // Layout

    // Create a descriptor set for shader constnats.
    // @API[descriptors]: here we need more granularity. The all in one helper is nice
    // but I would like to also be able to share layout and pools.
    Array<gfx::DescriptorSet> descriptor_sets(window.frames.length);
    for(usize i = 0; i < window.frames.length; i++) {
        vkr = gfx::CreateDescriptorSet(&descriptor_sets[i], vk, {
            .entries {
                {
                    .count = 1,
                    .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                },
            },
        });
    }
    assert(vkr == VK_SUCCESS);

    gfx::GraphicsPipeline pipeline = {};
    vkr = gfx::CreateGraphicsPipeline(&pipeline, vk, {
        .stages = {
            {
                .shader = vertex_shader,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
            },
            {
                .shader = fragment_shader,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            },
        },
        .vertex_bindings = {
            {
                .binding = 0,
                .stride = sizeof(glm::vec3),
            },
        },
        .vertex_attributes = {
            {
                .location = 0,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
            },
        },
        .depth = {
            .test = true,
            .write = true,
            .op = VK_COMPARE_OP_LESS,
            .format = VK_FORMAT_D32_SFLOAT,
        },
        .descriptor_sets = {
           descriptor_sets[0].layout,
        },
        .attachments = {
            {
                .format = window.swapchain_format,
            },
        },
    });
    assert(vkr == VK_SUCCESS);

    Array<gfx::Buffer> uniform_buffers(window.frames.length);
    for (usize i = 0; i < uniform_buffers.length; i++) {
        vkr = gfx::CreateBuffer(&uniform_buffers[i], vk, sizeof(Constants), {
            .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            .alloc = gfx::AllocPresets::DeviceMapped,
        });
        assert(vkr == VK_SUCCESS);

        gfx::WriteBufferDescriptor(descriptor_sets[i].set, vk, {
            .buffer = uniform_buffers[i].buffer,
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .binding = 0,
            .element = 0,
        });
    }

    gfx::Image depth_buffer;
    vkr = gfx::CreateImage(&depth_buffer, vk, {
        .width = window.fb_width,
        .height = window.fb_height,
        .format = VK_FORMAT_D32_SFLOAT,
        .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        .alloc = gfx::AllocPresets::DeviceDedicated,
    });
    assert(vkr == VK_SUCCESS);

    Array<VkSemaphore> copy_done_semaphores(window.frames.length);
    Array<VkSemaphore> render_done_semaphores(window.frames.length);
    for(usize i = 0; i < window.frames.length; i++) {
        gfx::CreateGPUSemaphore(vk.device, &copy_done_semaphores[i]);
        gfx::CreateGPUSemaphore(vk.device, &render_done_semaphores[i]);
    }

    App app = {};
    app.wait_for_events = true;
    app.frame_times.resize(ArrayCount(app.frame_times.data));
    app.last_frame_timestamp = platform::GetTimestamp();
    app.vertex_buffers_upload = move(vertex_buffers_upload);
    app.vertex_buffers_gpu = move(vertex_buffers_gpu);
    app.index_buffer = index_buffer;
    app.pipeline = pipeline.pipeline;
    app.descriptor_sets = move(descriptor_sets);
    app.uniform_buffers = move(uniform_buffers);
    app.layout = pipeline.layout;
    app.depth_buffer = depth_buffer;

    app.num_playback_frames = (u32)N;
    app.num_vertices = (u32)V;
    app.num_indices = (u32)I;
    app.mesh_file = file;
    app.mesh_stream = move(mesh_stream);

    app.copy_done_semaphores = move(copy_done_semaphores);
    app.render_done_semaphores = move(render_done_semaphores);

    auto Draw = [&app, &vk, &window] () {
        if (app.closed) return;

        platform::Timestamp timestamp = platform::GetTimestamp();
        float dt = (float)platform::GetElapsed(app.last_frame_timestamp, timestamp);
        app.last_frame_timestamp = timestamp;
        if (isnan(dt) || isinf(dt)) {
            dt = 0.0f;
        }
        app.frame_times[app.current_frame % app.frame_times.length] = dt;

        float avg_frame_time = 0.0f;
        for (usize i = 0; i < app.frame_times.length; i++) {
            avg_frame_time += app.frame_times[i];
        }
        avg_frame_time /= (f32)app.frame_times.length;

        gfx::SwapchainStatus swapchain_status = gfx::UpdateSwapchain(&window, vk);
        if (swapchain_status == gfx::SwapchainStatus::FAILED) {
            printf("Swapchain update failed\n");
            exit(1);
        }
        app.force_swapchain_update = false;

        if (swapchain_status == gfx::SwapchainStatus::MINIMIZED) {
            app.wait_for_events = true;
            return;
        }
        else if(swapchain_status == gfx::SwapchainStatus::RESIZED) {
            // Resize framebuffer sized elements.
            gfx::DestroyImage(&app.depth_buffer, vk);
            VkResult vkr = gfx::CreateImage(&app.depth_buffer, vk, {
                .width = window.fb_width,
                .height = window.fb_height,
                .format = VK_FORMAT_D32_SFLOAT,
                .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                .alloc = gfx::AllocPresets::DeviceDedicated,
            });
            if(vkr != VK_SUCCESS) {
                printf("Depth buffer resize failed\n");
                exit(1);
            }
        }

        app.wait_for_events = false;

        // Acquire current frame
        gfx::Frame& frame = gfx::WaitForFrame(&window, vk);
        gfx::Result ok = gfx::AcquireImage(&frame, &window, vk);
        if (ok != gfx::Result::SUCCESS) {
            return;
        }

        // Playback update
        if (app.playback_enabled) {
            app.playback_delta += dt;
            u32 frames_to_step = (u32)(app.playback_delta * 25.0f);
            app.playback_frame = (app.playback_frame + frames_to_step + app.num_playback_frames) % app.num_playback_frames;
            app.playback_delta -= frames_to_step * (1.0f / 25.0f);
        }


        // ImGui
        gui::BeginFrame();

        ImGui::DockSpaceOverViewport(0, NULL, ImGuiDockNodeFlags_PassthruCentralNode);

        if (ImGui::Begin("Playback")) {
#if 1
            struct Getter {
                static float fn(void* data, int index) {
                    App* a = (App*)data;

                    usize i = (index - (a->current_frame % a->frame_times.length) + a->frame_times.length) % a->frame_times.length;
                    return 1.0f / a->frame_times[a->frame_times.length - i - 1];
                }
            };
            //ImGui::PlotLines("", app.frame_times.data, (int)app.frame_times.length, 0, 0, 0.0f, .0f, ImVec2(100, 30));
            ImGui::PlotLines("##fps", Getter::fn, &app, (int)app.frame_times.length, 0, 0, 0.0f, 600.0f, ImVec2(100, 30));
            ImGui::SameLine();
            ImGui::Text("FPS: %.2f (%.2fms) [%.2f (%.2fms)]", 1.0f / dt, dt * 1.0e3f, 1.0 / avg_frame_time, avg_frame_time * 1.0e3f);

            int frame = app.playback_frame;
            if (ImGui::SliderInt("Playback", &frame, 0, app.num_playback_frames - 1)) {
                app.playback_frame = frame;
                app.playback_delta = 0.0f;
            }

            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            ImVec2 p = ImGui::GetCursorScreenPos();
            f32 x = p.x + 2.0f;
            f32 y = p.y + 4.0f;
            f32 width = 4.0f;
            f32 height = 16.0f;
            for (u32 i = 0; i < app.num_playback_frames; i++) {
                // Normalize frame index
                u64 frame_index = i;
                if (frame_index < app.mesh_stream.stream_cursor) {
                    frame_index += app.mesh_stream.stream_length;
                }

                // If in bounds of the buffer
                if (frame_index < app.mesh_stream.stream_cursor + app.mesh_stream.buffer.length) {
                    u64 delta = frame_index - app.mesh_stream.stream_cursor;
                    u64 buffer_index = (app.mesh_stream.buffer_offset + delta) % app.mesh_stream.buffer.length;

                    u32 color = 0xFF000000;

                    BufferedStream<FileReadWork>::EntryState state = app.mesh_stream.buffer[buffer_index].state.load(std::memory_order_relaxed);
                    switch (state) {
                    case BufferedStream<FileReadWork>::EntryState::Empty: color = 0xFF000000; break;
                    case BufferedStream<FileReadWork>::EntryState::Filling: color = 0xFF00FFFF; break;
                    case BufferedStream<FileReadWork>::EntryState::Canceling: color = 0xFF0000FF; break;
                    case BufferedStream<FileReadWork>::EntryState::Done: color = 0xFF00FF00; break;
                    }
                    draw_list->AddRectFilled(ImVec2(x, y), ImVec2(x + width, y + height), color);
                }

                u32 outline_color = 0xFFFFFFFF;
                if (app.mesh_stream.stream_cursor == i) {
                    outline_color = 0xFF000000;
                }
                draw_list->AddRect(ImVec2(x, y), ImVec2(x + width, y + height), outline_color, 0, 0, 1.0f);


                x += width + 1.0f;
            }
#endif
        }
        ImGui::End();
        ImGui::ShowDemoWindow();

        gui::EndFrame();

        // Reset command pool
        gfx::BeginCommands(frame.command_pool, frame.command_buffer, vk);

        //u32 animation_frame = app.current_frame % app.num_frames;
        u32 size = app.num_vertices * sizeof(vec3);

    #if 0
        Array<vec3> vertices(app.num_vertices);
        ReadAtOffset(app.mesh_file, vertices.as_bytes(), 12 + (u64)size * animation_frame);
        memcpy(app.vertex_map.data + buffer_offset, vertices.data, size);
    #else
    #if DIRECT_UPLOAD
        platform::FileReadWork w = app.mesh_stream.get_frame(app.playback_frame);
        platform::Timestamp begin = platform::GetTimestamp();
        memcpy(app.vertex_buffers_gpu[app.frame_index].map.data, w.buffer.data, size);
        double elapsed = platform::GetElapsed(begin, platform::GetTimestamp());
        // logging::info("sequence", "memcpy to device: %6.3fms (%6.3fGB/s) (%6.3f MB)", elapsed * 1000, (double)size / elapsed / (1024 * 1024 * 1024), (double)size / (1024 * 1024));
    #else
        platform::FileReadWork w = app.mesh_stream.get_frame(app.playback_frame);
        // platform::Timestamp begin = platform::GetTimestamp();
        // memcpy(app.vertex_buffers_upload[app.frame_index].map.data, w.buffer.data, size);
        // double elapsed = platform::GetElapsed(begin, platform::GetTimestamp());
        // logging::info("sequence", "memcpy to host: %6.3fms (%6.3fGB/s) (%6.3f MB)", elapsed * 1000, (double)size / elapsed / (1024 * 1024 * 1024), (double)size / (1024 * 1024));
    #endif
    #endif

#if DIRECT_UPLOAD
        // Flush caches on the CPU if the memory is not coherent (make available)
        VkMemoryPropertyFlags properties = {};
        vmaGetAllocationMemoryProperties(vk.vma, app.vertex_buffers_gpu[app.frame_index].allocation, &properties);
        if(!(properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            VmaAllocationInfo info = {};
            vmaGetAllocationInfo(vk.vma, app.vertex_buffers_gpu[app.frame_index].allocation, &info);

            VkMappedMemoryRange range = { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE };
            range.memory = info.deviceMemory;
            range.offset = 0;
            range.size = VK_WHOLE_SIZE;
            vkFlushMappedMemoryRanges(vk.device, 1, &range);
        }
#else
        if(vk.copy_queue != VK_NULL_HANDLE) {
            gfx::BeginCommands(frame.copy_command_pool, frame.copy_command_buffer, vk);
            VkBufferCopy region;
            region.srcOffset = 0;
            region.dstOffset = 0;
            region.size = size;
            vkCmdCopyBuffer(frame.copy_command_buffer, app.vertex_buffers_upload[app.frame_index].buffer, app.vertex_buffers_gpu[app.frame_index].buffer, 1, &region);

            // Queue transfer on copy queue
            gfx::CmdBufferBarrier(frame.copy_command_buffer, {
                .src_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT_KHR,
                .src_access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .src_queue = vk.copy_queue_family_index,
                .dst_queue = vk.queue_family_index,
                .buffer = app.vertex_buffers_gpu[app.frame_index].buffer,
                .offset = 0,
                .size = VK_WHOLE_SIZE,
            });

            gfx::EndCommands(frame.copy_command_buffer);

            VkPipelineStageFlags stage_mask = VK_PIPELINE_STAGE_TRANSFER_BIT;

            VkSubmitInfo submit_info = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &frame.copy_command_buffer;
            submit_info.waitSemaphoreCount = app.current_frame >= app.render_done_semaphores.length ? 1 : 0;
            submit_info.pWaitSemaphores = &app.render_done_semaphores[app.frame_index];
            submit_info.pWaitDstStageMask = &stage_mask;
            submit_info.signalSemaphoreCount = 1;
            submit_info.pSignalSemaphores = &app.copy_done_semaphores[app.frame_index];
            VkResult vkr = vkQueueSubmit(vk.copy_queue, 1, &submit_info, VK_NULL_HANDLE);
            assert(vkr == VK_SUCCESS);

            // Queue transfer on Graphics queue
            gfx::CmdBufferBarrier(frame.command_buffer, {
                .dst_stage = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT,
                .dst_access = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,
                .src_queue = vk.copy_queue_family_index,
                .dst_queue = vk.queue_family_index,
                .buffer = app.vertex_buffers_gpu[app.frame_index].buffer,
            });
        } else {
            VkBufferCopy region;
            region.srcOffset = 0;
            region.dstOffset = 0;
            region.size = size;
            vkCmdCopyBuffer(frame.command_buffer, app.vertex_buffers_upload[app.frame_index].buffer, app.vertex_buffers_gpu[app.frame_index].buffer, 1, &region);
        }
#endif

        // Invalidate caches on the GPU (make visible)
        // NOTE: I dont' think any memory barrier is needed here,
        // because submitting to the queue already counts as one.
        gfx::CmdBarriers(frame.command_buffer, {
            // .memory = {
            //     {
            //         .src_stage = VK_PIPELINE_STAGE_2_HOST_BIT,
            //         .dst_stage = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT,
            //         .src_access = VK_ACCESS_2_HOST_WRITE_BIT,
            //         .dst_access = VK_ACCESS_2_MEMORY_READ_BIT,
            //     },
            //     {
            //         .src_stage = VK_PIPELINE_STAGE_2_NONE,
            //         .dst_stage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
            //         .src_access = 0,
            //         .dst_access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            //     },
            // },
            .image = {
                {
                    .src_stage = VK_PIPELINE_STAGE_2_NONE,
                    .src_access = 0,
                    .dst_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    .dst_access = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    .image = frame.current_image,
                },
            }
        });


        // Begin rendering.
        VkClearColorValue color = { 0.1f, 0.2f, 0.4f, 1.0f };
        gfx::CmdBeginRendering(frame.command_buffer, {
            .color = {
                {
                    .view = frame.current_image_view,
                    .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
                    .store_op = VK_ATTACHMENT_STORE_OP_STORE,
                    .clear = color,
                },
            },
            .depth = {
                .view = app.depth_buffer.view,
                .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .store_op = VK_ATTACHMENT_STORE_OP_STORE,
                .clear = 1.0f,
            },
            .width = window.fb_width,
            .height = window.fb_height,
        });

        vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app.pipeline);

        VkDeviceSize offsets[1] = { 0 };
        vkCmdBindVertexBuffers(frame.command_buffer, 0, 1, &app.vertex_buffers_gpu[app.frame_index].buffer, offsets);

        vkCmdBindIndexBuffer(frame.command_buffer, app.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

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

        vec3 camera_position = vec3(2, -8, 2) * 2.0f;
        vec3 camera_target = vec3(0, 0, 0);
        f32 fov = 45.0f;
        f32 ar = (f32)window.fb_width / (f32)window.fb_height;

        Constants* constants = app.uniform_buffers[app.frame_index].map.as_type<Constants>();
        constants->color = vec3(0.8, 0.8, 0.8);
        constants->transform = glm::perspective(fov, ar, 0.01f, 100.0f) * glm::lookAt(camera_position, camera_target, vec3(0, 1, 0));

        vkCmdBindDescriptorSets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app.layout, 0, 1, &app.descriptor_sets[app.frame_index].set, 0, 0);

        vkCmdDrawIndexed(frame.command_buffer, app.num_indices, 1, 0, 0, 0);
        gfx::CmdEndRendering(frame.command_buffer);

        gfx::CmdBeginRendering(frame.command_buffer, {
            .color = {
                {
                    .view = frame.current_image_view,
                    .load_op = VK_ATTACHMENT_LOAD_OP_LOAD,
                    .store_op = VK_ATTACHMENT_STORE_OP_STORE,
                    .clear = color,
                },
            },
            .width = window.fb_width,
            .height = window.fb_height,
        });
        gui::Render(frame.command_buffer);
        gfx::CmdEndRendering(frame.command_buffer);

        gfx::CmdImageBarrier(frame.command_buffer, {
            .src_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .src_access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .dst_stage = VK_PIPELINE_STAGE_2_NONE,
            .dst_access = 0,
            .old_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .new_layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .image = frame.current_image,
        });

        gfx::EndCommands(frame.command_buffer);

        VkResult vkr;
#if DIRECT_UPLOAD
        vkr = gfx::Submit(frame, vk, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
#else
        if(vk.copy_queue == VK_NULL_HANDLE) {
            vkr = gfx::Submit(frame, vk, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        } else {
            VkSemaphore wait_semaphores[] = {
                frame.acquire_semaphore,
                app.copy_done_semaphores[app.frame_index],
            };

            VkPipelineStageFlags wait_stages[] = {
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
            };

            VkSemaphore signal_semaphores[] = {
                frame.release_semaphore,
                app.render_done_semaphores[app.frame_index],
            };

            VkSubmitInfo submit_info = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &frame.command_buffer;
            submit_info.waitSemaphoreCount = 2;
            submit_info.pWaitSemaphores = wait_semaphores,
            submit_info.pWaitDstStageMask = wait_stages;
            submit_info.signalSemaphoreCount = 2;
            submit_info.pSignalSemaphores = signal_semaphores;
            vkr = vkQueueSubmit(vk.queue, 1, &submit_info, frame.fence);
        }
        assert(vkr == VK_SUCCESS);
#endif

        vkr = gfx::PresentFrame(&window, &frame, vk);
        assert(vkr == VK_SUCCESS);

        app.frame_index = (app.frame_index + 1) % window.images.length;
        app.current_frame += 1;
    };

    auto KeyEvent = [&app, &window](gfx::Key key, gfx::Action action, gfx::Modifiers mods) {
        if (action == gfx::Action::Press || action == gfx::Action::Repeat) {
            if (key == gfx::Key::Escape) {
                gfx::CloseWindow(window);
            }

            if (key == gfx::Key::Period) {
                app.playback_frame += 1;
                app.playback_frame = (app.playback_frame + app.num_playback_frames) % app.num_playback_frames;
            }

            if (key == gfx::Key::Comma) {
                app.playback_frame -= 1;
                app.playback_frame = (app.playback_frame + app.num_playback_frames) % app.num_playback_frames;
            }

            if (key == gfx::Key::Space) {
                app.playback_enabled = !app.playback_enabled;
                app.playback_delta = 0.0f;
            }
        }
    };

    gfx::SetWindowCallbacks(&window, {
        .key_event = KeyEvent,
        .draw = Draw,
    });


    gui::ImGuiImpl imgui_impl;
    gui::CreateImGuiImpl(&imgui_impl, window, vk, {});

    while (true) {
        gfx::ProcessEvents(app.wait_for_events);

        if (gfx::ShouldClose(window)) {
            logging::info("sequence", "Window closed");
            app.closed = true;
            break;
        }

        // Draw
        Draw();
    };

    // Wait
    gfx::WaitIdle(vk);

    pool.destroy();

    gfx::DestroyImage(&app.depth_buffer, vk);

    for (usize i = 0; i < app.descriptor_sets.length; i++) {
        gfx::DestroyDescriptorSet(&app.descriptor_sets[i], vk);
    }

    for (usize i = 0; i < app.uniform_buffers.length; i++) {
        gfx::DestroyBuffer(&app.uniform_buffers[i], vk);
    }

    for (usize i = 0; i < window.frames.length; i++) {
        gfx::DestroyGPUSemaphore(vk.device, &app.render_done_semaphores[i]);
        gfx::DestroyGPUSemaphore(vk.device, &app.copy_done_semaphores[i]);
        gfx::DestroyBuffer(&app.vertex_buffers_gpu[i], vk);
    }
    #if !DIRECT_UPLOAD
    for (usize i = 0; i < BUFFER_SIZE; i++) {
        gfx::DestroyBuffer(&app.vertex_buffers_upload[i], vk);
    }
    #endif

    gfx::DestroyBuffer(&index_buffer, vk);

    gfx::DestroyShader(&vertex_shader, vk);
    gfx::DestroyShader(&fragment_shader, vk);

    gfx::DestroyGraphicsPipeline(&pipeline, vk);

    // Gui
    gui::DestroyImGuiImpl(&imgui_impl, vk);

    // Window
    gfx::DestroyWindowWithSwapchain(&window, vk);

    // Context
    gfx::DestroyContext(&vk);
}