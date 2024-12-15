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
using glm::vec3;
using glm::vec4;
using glm::mat4;
#include "types.h"

// TODO:
// [ ] Load data from disk
// [ ] First load everything from disk and decompress into chunks and only deal with descriptors
// [ ] Display load state in some sort of minimap in imgui
// [ ] Threaded loading
// [ ] Add framegraph
// [ ] Reimplement with minimal example
// [ ] Mips on last chunk
// [ ] Non pow2 images and non multiple of chunk size

struct App {
    gfx::Context* vk;
    gfx::Window* window;

    // Window stuff
    bool wait_for_events;
    bool closed;

    // Application data
    platform::Timestamp last_frame_timestamp;
    u64 current_frame;  // Total frame index, always increasing

    // Rendering
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSet descriptor_set;
    gfx::Image image;   // Application backbuffer (blitted to swapchain backbuffer)
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
                .count = 1,
                .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            },
            {
                .count = 1,
                .type = VK_DESCRIPTOR_TYPE_SAMPLER,
            },
            {
                .count = 1024,
                .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            },
        }
    });
    if (result != gfx::Result::SUCCESS) {
        logging::error("bigimage", "Failed to create descriptor set\n");
        exit(100);
    }

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

    gfx::Sampler linear_sampler;
    vkr = gfx::CreateSampler(&linear_sampler, vk, {
        .min_filter = VK_FILTER_LINEAR,
        .mag_filter = VK_FILTER_LINEAR,
        .mipmap_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
    });
    if (result != gfx::Result::SUCCESS) {
        logging::error("bigimage", "Failed to create sampler\n");
        exit(100);
    }

    gfx::WriteSamplerDescriptor(bindless.set, vk, {
        .sampler = linear_sampler.sampler,
        .binding = 1,
        .element = 0,
    });

    // Pipeline
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
        .descriptor_sets = {
            bindless.layout,
        },
    });
    if (result != gfx::Result::SUCCESS) {
        logging::error("bigimage", "Failed to create compute pipeline\n");
        exit(100);
    }

    // Read image
    Array<u8> zmip_data = platform::ReadEntireFile("N:\\scenes\\hubble\\heic0601a.zmip");
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

    Array<u8> last_interleaved(header.chunk_width * header.chunk_height * header.channels);
    for (usize l = 0; l < header.levels; l++) {
        usize chunks_x = ((header.width >> l) + header.chunk_width - 1) / header.chunk_width;
        usize chunks_y = ((header.height >> l) + header.chunk_height - 1) / header.chunk_height;
        levels.add(zmip.consume_view<ZMipBlock>(chunks_y * chunks_x));

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
                if ((l + 1 == header.levels) && x == 0 && y == 0) {
                    ArrayView<u8> chunk = ArrayView<u8>(zmip_data).slice(b.offset, b.size);
                    usize frame_size = ZSTD_getFrameContentSize(chunk.data, chunk.length);
                    if (frame_size != last_interleaved.length) {
                        logging::error("bigimage/parse/chunk", "Compressed chunk frame size %llu does not match expected size %llu", frame_size, last_interleaved.length);
                        exit(102);
                    }
                    ZSTD_decompress(last_interleaved.data, last_interleaved.length, chunk.data, chunk.length);
                }
            }
        }
    }

    // Undo delta coding
    for (usize i = 1; i < last_interleaved.length; i++) {
        last_interleaved[i] += last_interleaved[i - 1];
    }

    // Deinterleave planes and add alpha
    Array<u8> last(header.chunk_width * header.chunk_height * 4);
    for (usize y = 0; y < header.chunk_height; y++) {
        for (usize x = 0; x < header.chunk_width; x++) {
            for (usize c = 0; c < header.channels; c++) {
                last[(y * header.chunk_width + x) * 4 + c] = last_interleaved[((header.chunk_height * c + y) * header.chunk_width) + x];
            }
            last[(y * header.chunk_width + x) * 4 + 3] = 255;
        }
    }

    // Upload image
    gfx::Image last_mip = {};
    gfx::CreateAndUploadImage(&last_mip, vk, last, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, {
        .width = header.chunk_width,
        .height = header.chunk_height,
        .format = VK_FORMAT_R8G8B8A8_SRGB,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .memory_required_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    });

    gfx::WriteImageDescriptor(bindless.set, vk, {
        .view = last_mip.view,
        .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .binding = 2,
        .element = 0,
    });

    // USER: application
    App app = {};
    app.window = &window;
    app.vk = &vk;
    app.wait_for_events = false;
    app.last_frame_timestamp = platform::GetTimestamp();
    app.pipeline = pipeline.pipeline;
    app.layout = pipeline.layout;
    app.descriptor_set = bindless.set;
    app.image = image;

    auto Draw = [&app, &vk, &window, &bindless] () {
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
        else if(swapchain_status == gfx::SwapchainStatus::RESIZED) {
            // USER: resize (e.g. framebuffer sized elements)
            gfx::DestroyImage(&app.image, vk);

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
        }

        // Acquire current frame
        gfx::Frame& frame = gfx::WaitForFrame(&window, vk);
        gfx::Result ok = gfx::AcquireImage(&frame, &window, vk);
        if (ok != gfx::Result::SUCCESS) {
            return;
        }

        {
            gui::BeginFrame();

            // USER: gui
            gui::DrawStats(dt, window.fb_width, window.fb_height);

            //ImGui::ShowDemoWindow();
            int32_t texture_index = 0;
            if (ImGui::Begin("Editor")) {
                ImGui::SliderInt("Color", &texture_index, 0, 3);
            }
            ImGui::End();

            gui::EndFrame();
        }

        {
            gfx::BeginCommands(frame.command_pool, frame.command_buffer, vk);

            // USER: draw commands
            VkClearColorValue color = { 0.1f, 0.2f, 0.4f, 1.0f };

            gfx::CmdImageBarrier(frame.command_buffer, app.image.image, {
                .src_stage  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .dst_stage  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .src_access = VK_ACCESS_2_TRANSFER_READ_BIT,
                .dst_access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
                .new_layout = VK_IMAGE_LAYOUT_GENERAL,
            });

            vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, app.pipeline);
            vkCmdBindDescriptorSets(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, app.layout, 0, 1, &app.descriptor_set, 0, 0);
            vkCmdDispatch(frame.command_buffer, (window.fb_width + 7) / 8, (window.fb_height + 7) / 8, 1);

            gfx::CmdImageBarrier(frame.command_buffer, app.image.image, {
                .src_stage  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dst_stage  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .src_access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                .dst_access = VK_ACCESS_2_TRANSFER_READ_BIT,
                .old_layout = VK_IMAGE_LAYOUT_GENERAL,
                .new_layout = VK_IMAGE_LAYOUT_GENERAL,
            });

            gfx::CmdImageBarrier(frame.command_buffer, frame.current_image, {
                .src_stage  = VK_PIPELINE_STAGE_2_NONE,
                .dst_stage  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
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
                .src_stage  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .dst_stage  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .src_access = 0,
                .dst_access = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                .old_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            });

            VkRenderingAttachmentInfo attachment_info = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
            attachment_info.imageView = frame.current_image_view;
            attachment_info.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            attachment_info.resolveMode = VK_RESOLVE_MODE_NONE;
            attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
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

            ImDrawData* draw_data = ImGui::GetDrawData();
            ImGui_ImplVulkan_RenderDrawData(draw_data, frame.command_buffer);

            vkCmdEndRenderingKHR(frame.command_buffer);

            gfx::CmdImageBarrier(frame.command_buffer, frame.current_image, {
                .src_stage  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .dst_stage  = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
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
    };

    gfx::SetWindowCallbacks(&window, {
        .draw = Draw,
        // .key_event = KeyEvent,
        // .mouse_event = MouseEvent,
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
    gfx::DestroyImage(&last_mip, vk);
    gfx::DestroyImage(&app.image, vk);
    gfx::DestroySampler(&linear_sampler, vk);
    gfx::DestroyShader(&shader, vk);
    gfx::DestroyComputePipeline(&pipeline, vk);
    gfx::DestroyBindlessDescriptorSet(&bindless, vk);

    // Gui
    gui::DestroyImGuiImpl(&imgui_impl, vk);

    // Window
    gfx::DestroyWindowWithSwapchain(&window, vk);

    // Context
    gfx::DestroyContext(&vk);
}
