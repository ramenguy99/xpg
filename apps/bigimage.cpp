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
// [ ] Implement view choice on the CPU and update descriptors
//     [ ] Likely use instanced quads or single buffer of tris (quad count is gonna be super low, so anything works here)
//     [ ] Zooming and panning
// [ ] Display load state in some sort of minimap in imgui
// [ ] Fix delta coding (likely a signed vs unsigned either here or in python)
// [ ] Threaded loading
// [ ] Add framegraph
// [ ] Reimplement with minimal example
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
                .count = 1,
                .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            },
            {
                .count = 1,
                .type = VK_DESCRIPTOR_TYPE_SAMPLER,
            },
            {
                .count = 1024 * 1024,
                .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            },
            {
                .count = (u32)window.frames.length,
                .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            },
        }
        });
    if (result != gfx::Result::SUCCESS) {
        logging::error("bigimage", "Failed to create descriptor set\n");
        exit(100);
    }

#if 0
    gfx::Image image = {};
    gfx::CreateImage(&image, vk, {
        .width = window.fb_width,
        .height = window.fb_height,
        .format = window.swapchain_format,
        .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .alloc_flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        .memory_required_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        });

    gfx::WriteImageDescriptor(bindless.set, vk, {
        .view = image.view,
        .layout = VK_IMAGE_LAYOUT_GENERAL,
        .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .binding = 0,
        .element = 0,
        });
#endif

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
#if 0
    Array<u8> code = platform::ReadEntireFile("res/bigimage.comp.spirv");
    gfx::Shader shader = {};
    vkr = gfx::CreateShader(&shader, vk, code);
    if (result != gfx::Result::SUCCESS) {
        logging::error("bigimage", "Failed to create shader\n");
        exit(100);
    }

    gfx::ComputePipeline pipeline = {};
    vkr = gfx::CreateComputePipeline(&pipeline, vk, {
        .shader = shader,
        .entry = "main",
        .push_constants = {
            {
                .flags = VK_SHADER_STAGE_COMPUTE_BIT,
                .offset = 0,
                .size = 4,
            },
        },
        .descriptor_sets = {
            bindless.layout,
        },
        });
    if (result != gfx::Result::SUCCESS) {
        logging::error("bigimage", "Failed to create compute pipeline\n");
        exit(100);
    }
#else
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
                .size = 28,
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
#endif

    // Read image
    Array<u8> zmip_data = platform::ReadEntireFile("N:\\scenes\\hubble\\heic0601a_test.zmip");
    // Array<u8> zmip_data = platform::ReadEntireFile("N:\\scenes\\hubble\\heic0601a_nodiff.zmip");
    // Array<u8> zmip_data = platform::ReadEntireFile("N:\\scenes\\hubble\\heic0601a_256_nodiff.zmip");
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

                // // Undo delta coding
                // for (usize i = 1; i < interleaved.length; i++) {
                //     interleaved[i] += interleaved[i - 1];
                // }

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
        // Window stuff
        bool wait_for_events;
        bool closed;

        // Application data
        platform::Timestamp last_frame_timestamp;
        u64 current_frame;  // Total frame index, always increasing
        Array<Chunk> chunks;
        vec2 offset;
        s32 zoom;
        bool first_frame_done;

        // Rendering
        VkPipeline pipeline;
        VkPipelineLayout layout;
        VkDescriptorSet descriptor_set;
    #if 0
        gfx::Image image;           // Application backbuffer (blitted to swapchain backbuffer)
    #endif
        Array<gfx::Buffer> chunks_buffers; // Buffer containing chunk metadata, one per frame in flight
        gfx::Buffer vertex_buffer;
        s32 descriptor_index;
        s32 descriptor_count;
        u32 frame_index; // Rendering frame index, wraps around at the number of frames in flight
    };

    // USER: application
    App app = {};
    app.wait_for_events = false;
    app.last_frame_timestamp = platform::GetTimestamp();
    app.pipeline = pipeline.pipeline;
    app.layout = pipeline.layout;
    app.descriptor_set = bindless.set;
#if 0
    app.image = image;
#endif
    app.descriptor_index = (s32)all_images.length - 1;
    app.descriptor_count = (s32)all_images.length;
    app.chunks_buffers = Array<gfx::Buffer>(window.frames.length);
    app.chunks = Array<Chunk>();
    app.offset = vec2(0, 0);
    app.zoom = 0;
    app.frame_index = 0;
    app.vertex_buffer = vertex_buffer;

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
#if 0
            VkResult vkr = gfx::CreateImage(&app.image, vk, {
                .width = window.fb_width,
                .height = window.fb_height,
                .format = window.swapchain_format,
                .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                .alloc_flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
                .memory_required_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                });

            if (vkr != VK_SUCCESS) {
                logging::error("bigimage/draw", "Failed to resize image");
            }

            {
                VkDescriptorImageInfo desc_info = {};
                desc_info.imageView = app.image.view;
                desc_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                VkWriteDescriptorSet write_descriptor_set = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                write_descriptor_set.dstSet = bindless.set;
                write_descriptor_set.dstArrayElement = 0; // Element 0 in the set
                write_descriptor_set.descriptorCount = 1;
                write_descriptor_set.pImageInfo = &desc_info;
                write_descriptor_set.dstBinding = 0; // Here we use 0 because in our descriptor bindings, we have STORAGE_IMAGE at index 0
                write_descriptor_set.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;

                // Actually write the descriptor to the GPU visible heap
                vkUpdateDescriptorSets(vk.device, 1, &write_descriptor_set, 0, nullptr);
            }
#endif
            
            
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
                        .binding = 3,
                        .element = (u32)i,
                    });
                }
            }
        }

        // Chunks
        ivec2 offset = app.offset;
        ivec2 view_size = ivec2(window.fb_width, window.fb_height);

        ivec2 img_size = ivec2(header.width, header.height);
        ivec2 chunk_size = ivec2(header.chunk_width, header.chunk_height);
        ivec2 chunks = (img_size + chunk_size - ivec2(1, 1)) / chunk_size;
        s32 chunks_per_row = chunks.x;

        // ivec2 min_pixel = glm::max(-offset, 0);
        // ivec2 max_pixel = glm::min(min_pixel + view_size, img_size);

        app.chunks.length = 0;
        for (s32 y = 0; y < img_size.y; y += chunk_size.y) {
            for (s32 x = 0; x < img_size.x; x += chunk_size.x) {
                ivec2 img_coords = ivec2(x, y) - offset;

                // Skip chunks that are of bounds of the image
                if (img_coords.x + chunk_size.x - 1 < 0 || img_coords.x /* - chunk_size.x */ >= img_size.x ||
                    img_coords.y + chunk_size.y - 1 < 0 || img_coords.y /* - chunk_size.y */ >= img_size.y) {
                    continue;
                }

                ivec2 chunk = img_coords / chunk_size;
                assert(chunk.x >= 0 && chunk.x < chunks.x);
                assert(chunk.y >= 0 && chunk.y < chunks.y);

                Chunk c = {};
                c.desc_index = chunk.y * chunks_per_row + chunk.x + level_chunk_offsets[app.zoom];
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
                ImGui::InputInt("Index", &app.descriptor_index);
                app.descriptor_index = Clamp(app.descriptor_index, 0, app.descriptor_count - 1);

                ImGui::InputInt("Zoom", &app.zoom);
                app.zoom = Clamp(app.zoom, 0, (s32)level_chunk_offsets.length - 1);

                ImGui::DragFloat2("Offset", &app.offset.x);
            }
            ImGui::End();

            gui::EndFrame();
        }

        {
            gfx::BeginCommands(frame.command_pool, frame.command_buffer, vk);

            // USER: draw commands
            VkClearColorValue color = { 0.1f, 0.2f, 0.4f, 1.0f };

#if 0
            gfx::CmdImageBarrier(frame.command_buffer, app.image.image, {
                .src_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .dst_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .src_access = VK_ACCESS_2_TRANSFER_READ_BIT,
                .dst_access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
                .new_layout = VK_IMAGE_LAYOUT_GENERAL,
                });

            vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, app.pipeline);
            vkCmdPushConstants(frame.command_buffer, app.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &app.descriptor_index);
            vkCmdBindDescriptorSets(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, app.layout, 0, 1, &app.descriptor_set, 0, 0);
            vkCmdDispatch(frame.command_buffer, (window.fb_width + 7) / 8, (window.fb_height + 7) / 8, 1);

            gfx::CmdImageBarrier(frame.command_buffer, app.image.image, {
                .src_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dst_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .src_access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                .dst_access = VK_ACCESS_2_TRANSFER_READ_BIT,
                .old_layout = VK_IMAGE_LAYOUT_GENERAL,
                .new_layout = VK_IMAGE_LAYOUT_GENERAL,
                });

            gfx::CmdImageBarrier(frame.command_buffer, frame.current_image, {
                .src_stage = VK_PIPELINE_STAGE_2_NONE,
                .dst_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .src_access = 0,
                .dst_access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
                .new_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                });

            VkImageCopy regions = {};
            regions.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            regions.srcSubresource.mipLevel = 0;
            regions.srcSubresource.baseArrayLayer = 0;
            regions.srcSubresource.layerCount = 1;
            regions.dstSubresource = regions.srcSubresource;
            regions.extent.width = window.fb_width;
            regions.extent.height = window.fb_height;
            regions.extent.depth = 1;
            vkCmdCopyImage(frame.command_buffer, app.image.image, VK_IMAGE_LAYOUT_GENERAL, frame.current_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &regions);

            gfx::CmdImageBarrier(frame.command_buffer, frame.current_image, {
                .src_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .dst_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .src_access = 0,
                .dst_access = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                .old_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                });

#else
            // gfx::CmdBufferBarrier(frame.command_buffer, , {
            //     .src_stage = VK_PIPELINE_STAGE_2_NONE,
            //     .dst_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            //     .src_access = 0,
            //     .dst_access = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            //     });

            gfx::CmdImageBarrier(frame.command_buffer, frame.current_image, {
                .src_stage = VK_PIPELINE_STAGE_2_NONE,
                .dst_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .src_access = 0,
                .dst_access = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
                .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                });

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
                vec2 offset;
                vec2 inv_window_size;
                u32 frame_id;
            };
            Constants constants = {
                .scale = vec2(chunk_size),
                .offset = vec2(0, 0),
                .inv_window_size = vec2(1.0f / (f32)window.fb_width, 1.0f / (f32)window.fb_height),
                .frame_id = (u32)app.frame_index,
            };
            vkCmdPushConstants(frame.command_buffer, app.layout, VK_SHADER_STAGE_ALL, 0, sizeof(Constants), &constants);

            vkCmdDraw(frame.command_buffer, 6, (u32)app.chunks.length, 0, 0);
#endif


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

    auto MouseButtonEvent = [&app](ivec2 pos, gfx::MouseButton button, gfx::Action action, gfx::Modifiers mods) {
        if (ImGui::GetIO().WantCaptureMouse) return;
    };

    auto MouseScrollEvent = [&app](ivec2 pos, ivec2 scroll, gfx::Modifiers mods) {
        if (ImGui::GetIO().WantCaptureMouse) return;
    };

    gfx::SetWindowCallbacks(&window, {
        .draw = Draw,
        .mouse_button_event = MouseButtonEvent,
        .mouse_scroll_event = MouseScrollEvent,
    });

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
#if 1
    gfx::DestroyShader(&vert_shader, vk);
    gfx::DestroyShader(&frag_shader, vk);
    gfx::DestroyGraphicsPipeline(&pipeline, vk);
#else
    gfx::DestroyImage(&app.image, vk);
    gfx::DestroyShader(&shader, vk);
    gfx::DestroyComputePipeline(&pipeline, vk);
#endif
    gfx::DestroyBindlessDescriptorSet(&bindless, vk);

    // Gui
    gui::DestroyImGuiImpl(&imgui_impl, vk);

    // Window
    gfx::DestroyWindowWithSwapchain(&window, vk);

    // Context
    gfx::DestroyContext(&vk);
}
