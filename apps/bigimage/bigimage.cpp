#include <xpg/gui.h>
#include <xpg/log.h>
#include <xpg/platform.h>

#include "zmip.h"
#include "chunk_cache.h"

using glm::vec2;
using glm::ivec2;
using glm::vec3;
using glm::vec4;
using glm::mat4;

using namespace xpg;

// TODO:
// [x] Load data from disk
// [x] First load everything from disk and decompress into chunks
// [x] Upload all chunks
// [x] Implement view choice on the CPU and update descriptors
//     [x] Likely use instanced quads or single buffer of tris (quad count is gonna be super low, so anything works here)
//     [x] Zooming and panning
// [x] Display load state in some sort of minimap in imgui
// [x] Fix delta coding (issue was delta per plane vs per whole chunk)
// [x] Full Sync:
//     [x] issue load and return desc synchronously
//     [x] bounded LRU cache with queue
//     [x] Cache resizing when screen is resized (ideally only incrementally add)
// [x] Threaded sync:  issue all needed loads to pool and wait for all of them to be satisfied before drawing.
// [ ] Threaded async: do loading and uploads on a different thread, maybe can have pool for loading / decompressing and worker thread / main thread doing uploads, but wait at the end or at transfer buffer exaustion (do something smart with semaphores/counters?)
//     [ ] Copy queue:     do uploads with copy queue, need to do queue transfer
// [ ] Prefetch:       after all loads are satisfied issue loads for neighbors
// [ ] Cancellation:   we can cancel neighbors from previous frame if they are not used / neighbors in this frame. (e.g. at zoom lvl 1 we prefetch lvl 0 and 2, if moving to 2 we can cancel prefetch at 0)
// [ ] Load all mode:  just increase the cache to the total size, issue all loads, done.
// [ ] Add framegraph?
// [ ] Reimplement with app::Application helper, compare code cut
// [ ] Mips on last chunk
// [ ] Non pow2 images and non multiple of chunk size

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s FILE\n", argv[0]);
        exit(1);
    }
    ZMipFile zmip = LoadZmipFile(argv[1]);

    gfx::Result result;
    result = gfx::Init();
    if (result != gfx::Result::SUCCESS) {
        logging::error("bigimage", "Failed to initialize platform\n");
    }

    gfx::Context vk = {};
    result = gfx::CreateContext(&vk, {
        .minimum_api_version = (u32)VK_API_VERSION_1_1,
        .required_features = gfx::DeviceFeatures::DYNAMIC_RENDERING | gfx::DeviceFeatures::DESCRIPTOR_INDEXING | gfx::DeviceFeatures::SYNCHRONIZATION_2,
        .enable_validation_layer = true,
        //        .enable_gpu_based_validation = true,
    });
    if (result != gfx::Result::SUCCESS) {
        logging::error("bigimage", "Failed to initialize vulkan\n");
        exit(100);
    }

    gfx::Window window = {};
    result = gfx::CreateWindowWithSwapchain(&window, vk, "XPG", 1600, 900);
    if (result != gfx::Result::SUCCESS) {
        logging::error("bigimage", "Failed to create vulkan window\n");
        exit(100);
    }

    VkResult vkr;

    // Descriptors
    gfx::DescriptorSet descriptor_set = {};
    vkr = gfx::CreateDescriptorSet(&descriptor_set, vk, {
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
                .count = 1024,
                .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            },
        },
        .flags = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT ||
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
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

    gfx::WriteSamplerDescriptor(descriptor_set.set, vk, {
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
            descriptor_set.layout,
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
            .alloc = gfx::AllocPresets::DeviceMapped,
        });
    assert(vkr == VK_SUCCESS);

    // Read image
    struct GpuChunk {
        vec2 position;
        u32 desc_index;
        u32 _padding;
    };

    struct App {
        // - Window
        bool wait_for_events = true;
        bool closed = false;
        bool first_frame_done = false;

        // - UI
        platform::Timestamp last_frame_timestamp;
        ivec2 drag_start_offset = ivec2(0, 0);
        ivec2 drag_start = ivec2(0, 0);
        bool dragging = false;
        vec2 offset = vec2(0, 0);
        s32 zoom = 0;
        s32 max_zoom = 0;
        bool show_grid = false;
        bool batched_chunk_upload = true;

        // - Bigimage
        Array<GpuChunk> gpu_chunks;
        ObjArray<Array<usize>> cpu_chunks;
        Array<ChunkId> batch_inputs;
        Array<u32> batch_outputs;
        usize total_max_chunks = 0;

        // - Rendering
        VkPipeline pipeline;
        VkPipelineLayout layout;
        VkDescriptorSet descriptor_set;
        Array<gfx::Buffer> chunks_buffers; // Buffer containing chunk metadata, one per frame in flight
        gfx::Buffer vertex_buffer;
        u32 frame_index = 0; // Rendering frame index, wraps around at the number of frames in flight
    };

    // USER: application
    App app = {};
    app.last_frame_timestamp = platform::GetTimestamp();
    app.pipeline = pipeline.pipeline;
    app.layout = pipeline.layout;
    app.descriptor_set = descriptor_set.set;
    app.chunks_buffers = Array<gfx::Buffer>(window.frames.length);
    app.cpu_chunks = ObjArray<Array<usize>>(window.frames.length);
    app.vertex_buffer = vertex_buffer;
    app.max_zoom = (s32)(zmip.levels.length - 1);

    ChunkCache cache(zmip, 0, 0, 8, window.frames.length, vk, descriptor_set);

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

    auto Draw = [&app, &vk, &window, &descriptor_set, &zmip, &cache]() {
        if (app.closed) return;

        platform::Timestamp timestamp = platform::GetTimestamp();
        float dt = (float)platform::GetElapsed(app.last_frame_timestamp, timestamp);
        app.last_frame_timestamp = timestamp;

        gfx::SwapchainStatus swapchain_status = gfx::UpdateSwapchain(&window, vk);
        if (swapchain_status == gfx::SwapchainStatus::FAILED) {
            logging::error("bigimage/draw", "Swapchain update failed\n");
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
                        .alloc = gfx::AllocPresets::DeviceMapped,
                    });
                    gfx::WriteBufferDescriptor(descriptor_set.set, vk, {
                        .buffer = app.chunks_buffers[i].buffer,
                        .binding = 0,
                        .element = (u32)i,
                    });
                }

                // Resize buffer for storing gpu data to transfer to the gpu each frame
                app.gpu_chunks.resize(app.total_max_chunks);

                // Release all the chunks in use on resize
                for (usize i = 0; i < app.cpu_chunks.length; i++) {
                    for (usize j = 0; j < app.cpu_chunks[i].length; j++) {
                        cache.release_chunk(app.cpu_chunks[i][j]);
                    }
                    app.cpu_chunks[i].length = 0;
                }

                // Resize buffers for chunks in flight
                for (usize i = 0; i < app.cpu_chunks.length; i++) {
                    app.cpu_chunks[i].grow(app.total_max_chunks);
                }

                // Resize cache
                cache.resize(total_max_chunks * window.frames.length, total_max_chunks, vk, descriptor_set);
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
        Array<usize>& cpu_chunks = app.cpu_chunks[app.frame_index];

        // Decrease refcount of old frame chunks
        for (usize i = 0; i < cpu_chunks.length; i++) {
            cache.release_chunk(cpu_chunks[i]);
        }

        // Reset chunks of the current frame.
        cpu_chunks.length = 0;
        app.gpu_chunks.length = 0;
        app.batch_inputs.length = 0;

        for (s32 y = 0; y < view_size.y + remainder.y; y += chunk_size.y) {
            for (s32 x = 0; x < view_size.x + remainder.x; x += chunk_size.x) {
                if (x < (s32)app.offset.x || y < (s32)app.offset.y) {
                    continue;
                }
                ivec2 image_coords = (ivec2(x, y) - ivec2(app.offset)) << app.zoom;
                ivec2 chunk = image_coords / z_chunk_size;

                // Skip chunks that are of bounds of the image
                if (!((chunk.x >= 0 && chunk.x < chunks.x) && (chunk.y >= 0 && chunk.y < chunks.y))) continue;

                ChunkId id((u32)chunk.x, (u32)chunk.y, (u32)app.zoom);

                u32 desc_index = UINT32_MAX;
                if (app.batched_chunk_upload) {
                    app.batch_inputs.add(id);
                }
                else {
                    desc_index = cache.request_chunk_sync(id, vk, descriptor_set);
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

        {
            gfx::BeginCommands(frame.command_pool, frame.command_buffer, vk);

            // USER: pre-gui, but with frame

            // Upload chunks
            if (app.batched_chunk_upload) {
                app.batch_outputs.resize(app.batch_inputs.length);
                cache.request_chunk_batch(app.batch_inputs, app.batch_outputs, vk, descriptor_set, frame.command_buffer, app.frame_index);
                for (usize i = 0; i < app.batch_outputs.length; i++) {
                    app.gpu_chunks[i].desc_index = app.batch_outputs[i];

                    // Check that the same chunk is not used twice
                    for (usize j = i + 1; j < app.batch_outputs.length; j++) {
                        assert(app.batch_outputs[i] != app.batch_outputs[j]);
                    }
                }
            }

            // Upload chunk info buffer
            {
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

                    ImGui::Checkbox("Batched upload", &app.batched_chunk_upload);

                    ImGui::Separator();

                    ImGui::Text("Image size: %zu x %zu", zmip.header.width, zmip.header.height);
                    ImGui::Text("Loaded chunks: %zu / %zu", cache.images.length, cache.chunks.length);
                    ImGui::Text("Memory: %.2f / %.2f MiB", cache.chunk_memory_size * cache.images.length / 1024.0 / 1024.0,  cache.chunk_memory_size * cache.chunks.length / 1024.0 / 1024.0);
                    ImGui::Text("File size: %.2f MiB", zmip.file.size / 1024.0 / 1024.0);

                    ImGui::Separator();

                    // Draw minimap
                    ImDrawList* draw_list = ImGui::GetWindowDrawList();
                    ImVec2 corner = ImGui::GetCursorScreenPos();
                    f32 size = 4.0f;
                    f32 stride = 5.0f;
                    ivec2 img_size = ivec2(zmip.header.width, zmip.header.height);                // In image pixels
                    for (s32 level = 0; level < (s32)zmip.levels.length; level++) {
                        ivec2 chunk_size = ivec2(zmip.header.chunk_width, zmip.header.chunk_height) << level;  // In image pixels
                        ivec2 chunks = (img_size + chunk_size - ivec2(1, 1)) / chunk_size;  // Total chunks in image at this zoom level
                        for (s32 y = 0; y < chunks.y; y++) {
                            for (s32 x = 0; x < chunks.x; x++) {
                                ChunkCache::Chunk& c = cache.get_chunk(ChunkId(x, y, level));
                                u32 color = 0xFF0000BB;
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

            // USER: draw commands
            gfx::CmdImageBarrier(frame.command_buffer, {
                .src_stage = VK_PIPELINE_STAGE_2_NONE,
                .src_access = 0,
                .dst_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .dst_access = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
                .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .image = frame.current_image,
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
            gui::Render(frame.command_buffer);

            vkCmdEndRenderingKHR(frame.command_buffer);

            gfx::CmdImageBarrier(frame.command_buffer, {
                .src_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .src_access = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                .dst_stage = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                .dst_access = 0,
                .old_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .new_layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                .image = frame.current_image,
                });

            gfx::EndCommands(frame.command_buffer);
        }

        VkResult vkr;
        vkr = gfx::Submit(frame, vk, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        assert(vkr == VK_SUCCESS);

        vkr = gfx::PresentFrame(&window, &frame, vk);
        assert(vkr == VK_SUCCESS);

        app.frame_index = (app.frame_index + 1) % (u32)window.frames.length;
    };

    gfx::SetWindowCallbacks(&window, {
        .mouse_move_event = MouseMoveEvent,
        .mouse_button_event = MouseButtonEvent,
        .mouse_scroll_event = MouseScrollEvent,
        .draw = Draw,
    });

    gui::ImGuiImpl imgui_impl;
    gui::CreateImGuiImpl(&imgui_impl, window, vk, {});

    while (true) {
        gfx::ProcessEvents(app.wait_for_events);

        if (gfx::ShouldClose(window)) {
            logging::info("bigimage", "Window closed");
            app.closed = true;
            break;
        }

        // Draw
        Draw();
    };

    // Wait
    gfx::WaitIdle(vk);

    // USER: cleanup
    cache.destroy_resources(vk);

    for (usize i = 0; i < app.chunks_buffers.length; i++) {
        gfx::DestroyBuffer(&app.chunks_buffers[i], vk);
    }
    gfx::DestroyBuffer(&vertex_buffer, vk);

    gfx::DestroySampler(&sampler, vk);
    gfx::DestroyShader(&vert_shader, vk);
    gfx::DestroyShader(&frag_shader, vk);
    gfx::DestroyGraphicsPipeline(&pipeline, vk);
    gfx::DestroyDescriptorSet(&descriptor_set, vk);

    // Gui
    gui::DestroyImGuiImpl(&imgui_impl, vk);

    // Window
    gfx::DestroyWindowWithSwapchain(&window, vk);

    // Context
    gfx::DestroyContext(&vk);
}
