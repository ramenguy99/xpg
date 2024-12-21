#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utility> // std::move
#include <functional> // std::function
#include <mutex>
#include <unordered_map>

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
// [ ] Add framegraph
// [ ] Reimplement with app::Application helper, compare code cut
// [ ] Mips on last chunk
// [ ] Non pow2 images and non multiple of chunk size

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
    Array<u8> vert_code = platform::ReadEntireFile("res/bigimage.vert.spirv");
    gfx::Shader vert_shader = {};
    vkr = gfx::CreateShader(&vert_shader, vk, vert_code);
    if (result != gfx::Result::SUCCESS) {
        logging::error("bigimage", "Failed to create vertex shader\n");
        exit(100);
    }

    Array<u8> frag_code = platform::ReadEntireFile("res/bigimage.frag.spirv");
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

    // Read image
    Array<u8> zmip_data = platform::ReadEntireFile("N:\\scenes\\hubble\\heic0601a_test_delta.zmip");
    // Array<u8> zmip_data = platform::ReadEntireFile("N:\\scenes\\hubble\\heic0601a_256.zmip");
    ArrayView<u8> zmip = zmip_data;

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

    struct ZMipBlock {
        u64 offset;
        u32 size;
    };

#pragma pack(pop)

    if (zmip.length < sizeof(ZMipHeader)) {
        logging::error("bigimage/parse", "File smaller than header");
        exit(102);
    }

    ZMipHeader header = zmip.consume<ZMipHeader>();
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

    Array<ArrayView<ZMipBlock>> levels;

    Array<u8> interleaved(header.chunk_width * header.chunk_height * header.channels);
    Array<u8> deinterleaved(header.chunk_width * header.chunk_height * 4);
    Array<gfx::Image> all_images;
    Array<u32> level_chunk_offsets;

    u32 chunk_offset = 0;
    for (usize l = 0; l < header.levels; l++) {
        usize chunks_x = ((header.width >> l) + header.chunk_width - 1) / header.chunk_width;
        usize chunks_y = ((header.height >> l) + header.chunk_height - 1) / header.chunk_height;
        levels.add(zmip.consume_view<ZMipBlock>(chunks_y * chunks_x));
        level_chunk_offsets.add(chunk_offset);
        chunk_offset += (u32)(chunks_x * chunks_y);

        for (usize y = 0; y < chunks_y; y++) {
            for (usize x = 0; x < chunks_x; x++) {
                ZMipBlock b = levels[l][y * chunks_x + x];
                if (b.offset + b.size < b.offset) {
                    logging::error("bigimage/parse/map", "offset + size overflow on chunk (%llu, %llu) at level %llu", x, y, l);
                    exit(102);
                }
                if (b.offset + b.size > zmip_data.length) {
                    logging::error("bigimage/parse/map", "offset + size out of bounds chunk (%llu, %llu) at level %llu", x, y, l);
                    exit(102);
                }

                // logging::info("bigimage/parse/map", "%llu | (%llu , %llu) %llu %u", l, x, y, b.offset, b.size);
                ArrayView<u8> chunk = ArrayView<u8>(zmip_data).slice(b.offset, b.size);
                usize frame_size = ZSTD_getFrameContentSize(chunk.data, chunk.length);
                if (frame_size != interleaved.length) {
                    logging::error("bigimage/parse/chunk", "Compressed chunk frame size %llu does not match expected size %llu", frame_size, interleaved.length);
                    exit(102);
                }
                ZSTD_decompress(interleaved.data, interleaved.length, chunk.data, chunk.length);

                // Undo delta coding
                usize plane_size = header.chunk_width * header.chunk_height;
                for (usize c = 0; c < header.channels; c++) {
                    for (usize i = 1; i < plane_size; i++) {
                        interleaved[i + plane_size * c] += interleaved[i - 1 + plane_size * c];
                    }
                }

                // Deinterleave planes and add alpha
                for (usize y = 0; y < header.chunk_height; y++) {
                    for (usize x = 0; x < header.chunk_width; x++) {
                        for (usize c = 0; c < header.channels; c++) {
                            deinterleaved[(y * header.chunk_width + x) * 4 + c] = interleaved[((header.chunk_height * c + y) * header.chunk_width) + x];
                        }
                        deinterleaved[(y * header.chunk_width + x) * 4 + 3] = 255;
                    }
                }

                // Upload image
                gfx::Image image = {};
                gfx::CreateAndUploadImage(&image, vk, deinterleaved, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, {
                    .width = header.chunk_width,
                    .height = header.chunk_height,
                    .format = VK_FORMAT_R8G8B8A8_UNORM,
                    .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    .memory_required_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                });

                gfx::WriteImageDescriptor(bindless.set, vk, {
                    .view = image.view,
                    .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    .binding = 2,
                    .element = (u32)all_images.length,
                });

                all_images.add(image);
            }
        }
    }

    struct Chunk {
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
        Array<Chunk> chunks;
        u64 current_frame = 0;  // Total frame index, always increasing
        vec2 offset = vec2(0, 0);
        s32 zoom = 0;
        s32 max_zoom = 0;
        bool first_frame_done = false;
        ivec2 drag_start_offset = ivec2(0, 0);
        ivec2 drag_start = ivec2(0, 0);
        bool dragging = false;
        bool show_grid = false;

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
    app.descriptor_count = (s32)all_images.length;
    app.chunks_buffers = Array<gfx::Buffer>(window.frames.length);
    app.chunks = Array<Chunk>();
    app.vertex_buffer = vertex_buffer;
    app.max_zoom = (s32)(level_chunk_offsets.length - 1);

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

    auto Draw = [&app, &vk, &window, &bindless, &header, &level_chunk_offsets]() {
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
            ivec2 img_size = ivec2(header.width, header.height);
            ivec2 chunk_size = ivec2(header.chunk_width, header.chunk_height);
            ivec2 max_chunks = (img_size + chunk_size - ivec2(1, 1)) / chunk_size + ivec2(1, 1);
            u64 total_max_chunks = (u64)max_chunks.x * (u64)max_chunks.y;

            if (total_max_chunks * sizeof(Chunk) >= app.chunks_buffers[0].size) {
                for (usize i = 0; i < app.chunks_buffers.length; i++) {
                    gfx::DestroyBuffer(&app.chunks_buffers[i], vk);
                    gfx::CreateBuffer(&app.chunks_buffers[i], vk, total_max_chunks * sizeof(Chunk), {
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
            }
        }

        // Chunks
        ivec2 offset = app.offset;                                          // In screen pixel
        ivec2 view_size = ivec2(window.fb_width, window.fb_height);         // In screen pixel
        ivec2 img_size = ivec2(header.width, header.height);                // In image pixels
        ivec2 chunk_size = ivec2(header.chunk_width, header.chunk_height);  // In image pixels

        // Zoom dependant
        ivec2 z_chunk_size = chunk_size << app.zoom;                            // In image pixels
        ivec2 chunks = (img_size + z_chunk_size - ivec2(1, 1)) / z_chunk_size;  // Total chunks in image at this zoom level
        ivec2 remainder = (chunk_size - (view_size - offset) % chunk_size);

        app.chunks.length = 0;
        for (s32 y = -offset.y; y < view_size.y - offset.y + remainder.y; y += chunk_size.y) {
            for (s32 x = -offset.x; x < view_size.x - offset.x + remainder.x; x += chunk_size.x) {
                ivec2 image_coords = ivec2(x, y) << app.zoom;
                ivec2 chunk = image_coords / z_chunk_size;

                // Skip chunks that are of bounds of the image
                if (!((chunk.x >= 0 && chunk.x < chunks.x) && (chunk.y >= 0 && chunk.y < chunks.y))) continue;

                Chunk c = {};
                c.desc_index = chunk.y * chunks.x + chunk.x + level_chunk_offsets[app.zoom];
                c.position = offset + chunk * chunk_size;

                app.chunks.add(c);
            }
        }

        // Acquire current frame
        gfx::Frame& frame = gfx::WaitForFrame(&window, vk);
        gfx::Result ok = gfx::AcquireImage(&frame, &window, vk);
        if (ok != gfx::Result::SUCCESS) {
            return;
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

            ArrayView<u8> map((u8*)addr, app.chunks.size_in_bytes());
            map.copy_exact(app.chunks.as_bytes());

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
                app.zoom = Clamp(app.zoom, 0, (s32)level_chunk_offsets.length - 1);

                ImGui::DragFloat2("Offset", &app.offset.x);

                ImGui::Checkbox("Show grid", &app.show_grid);

                ImGui::Separator();

                // Draw minimap
                ImDrawList* draw_list = ImGui::GetWindowDrawList();
                ImVec2 corner = ImGui::GetCursorScreenPos();
                f32 size = 5.0f;
                f32 stride = 7.0f;
                ivec2 img_size = ivec2(header.width, header.height);                // In image pixels
                for (s32 level = 0; level < (s32)level_chunk_offsets.length; level++) {
                    ivec2 chunk_size = ivec2(header.chunk_width, header.chunk_height) << level;  // In image pixels
                    ivec2 chunks = (img_size + chunk_size - ivec2(1, 1)) / chunk_size;  // Total chunks in image at this zoom level
                    for (s32 y = 0; y < chunks.y; y++) {
                        for (s32 x = 0; x < chunks.x; x++) {
                            draw_list->AddRectFilled(corner + ImVec2(x * stride, y * stride), corner + ImVec2(x * stride + size, y * stride + size), 0xFF0000FF);
                        }
                    }

                    if (level == app.zoom) {
                        for (usize i = 0; i < app.chunks.length; i++) {
                            Chunk& c = app.chunks[i];
                            usize chunk_index = (usize)c.desc_index - level_chunk_offsets[app.zoom];
                            usize x = chunk_index % chunks.x;
                            usize y = chunk_index / chunks.x;
                            draw_list->AddRectFilled(corner + ImVec2(x * stride, y * stride), corner + ImVec2(x * stride + size, y * stride + size), 0xFF00FF00);
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

            VkClearColorValue color = { 0.1f, 0.2f, 0.4f, 1.0f };
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

            vkCmdDraw(frame.command_buffer, 6, (u32)app.chunks.length, 0, 0);

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
    for (usize i = 0; i < all_images.length; i++) {
        gfx::DestroyImage(&all_images[i], vk);
    }

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
