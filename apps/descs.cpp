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

#define XPG_VERSION 0

#include "defines.h"
#include "result.h"
#include "array.h"
#include "platform.h"
#include "threading.h"
#include "gfx.h"
#include "imgui_impl.h"
#include "buffered_stream.h"
// #include "graph_simpler.h"
#include "graph.h"

#define SPECTRUM_USE_DARK_THEME
#include "imgui_spectrum.h"
#include "roboto-medium.h"

using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::mat4;
#include "types.h"

// TODO:
// - Cleanup:
//   - remove imgui_impl asserts
//   - add utilities for bindless descriptor management / generation checks
// - Features:
//   - start working on a c++ framegraph
//   - slang integration / reflection / hot reloading
//   - more python bindings
//   - file formats 
struct App {
    gfx::Context* vk;
    gfx::Window* window;

    // Window stuff
    bool wait_for_events;
    bool closed;

    //- Application data.
    u64 last_frame_timestamp;
    u64 current_frame;          // Total frame index, always increasing

    // Rendering
    gfx::Buffer vertex_buffer;
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSet descriptor_set;
};

void Draw(App* app) {
    if (app->closed) return;

    auto& vk = *app->vk;
    auto& window = *app->window;

    u64 timestamp = glfwGetTimerValue();

    float dt = (float)((double)(timestamp - app->last_frame_timestamp) / (double)glfwGetTimerFrequency());

    gfx::SwapchainStatus swapchain_status = UpdateSwapchain(&window, vk);
    if (swapchain_status == gfx::SwapchainStatus::FAILED) {
        printf("Swapchain update failed\n");
        exit(1);
    }
    else if (swapchain_status == gfx::SwapchainStatus::MINIMIZED) {
        return;
    }
    else if(swapchain_status == gfx::SwapchainStatus::RESIZED) {
        // Resize framebuffer sized elements.
    }

    // app->wait_for_events = false;

    // Acquire current frame
    gfx::Frame* opt_frame = gfx::AcquireNextFrame(&window, vk);
    if (!opt_frame) {
        return;
    }
    gfx::Frame& frame = *opt_frame;

    gui::BeginFrame();

    //ImGui::ShowDemoWindow();
    int32_t texture_index = 0;
    if (ImGui::Begin("Editor")) {
        ImGui::SliderInt("Color", &texture_index, 0, 3);
    }
    ImGui::End();


    gui::EndFrame();


    // Reset command pool
    VkResult vkr;
    vkr = vkResetCommandPool(vk.device, frame.command_pool, 0);
    assert(vkr == VK_SUCCESS);

    // Record commands
    VkCommandBufferBeginInfo begin_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkr = vkBeginCommandBuffer(frame.command_buffer, &begin_info);
    assert(vkr == VK_SUCCESS);
   
    vkResetFences(vk.device, 1, &frame.fence);

    VkClearColorValue color = { 0.1f, 0.2f, 0.4f, 1.0f };

    VkImageMemoryBarrier barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = frame.current_image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    
    barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    
    vkCmdPipelineBarrier(frame.command_buffer,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 
                         0, 0, 
                         0, 0, 
                         1, &barrier);

    // Begin rendering.
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

    vkCmdBeginRenderingKHR(frame.command_buffer, &rendering_info);

    vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app->pipeline);
    

    VkDeviceSize offsets[1] = { 0 };
    vkCmdBindVertexBuffers(frame.command_buffer, 0, 1, &app->vertex_buffer.buffer, offsets);

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
    
    // vkCmdBindDescriptorSets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app->layout, 0, 1, &app->descriptor_sets[app->frame_index], 0, 0);

    // Only needs to happen once per frame, all pipelines that share this layout will use this.
    vkCmdBindDescriptorSets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app->layout, 0, 1, &app->descriptor_set, 0, 0);
    
    vkCmdPushConstants(frame.command_buffer, app->layout, VK_SHADER_STAGE_ALL, 0, 4, &texture_index);

    vkCmdDraw(frame.command_buffer, 6, 1, 0, 0);
    
    vkCmdEndRenderingKHR(frame.command_buffer);
    
    attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    rendering_info.pDepthAttachment = 0;
    vkCmdBeginRenderingKHR(frame.command_buffer, &rendering_info);
    
    ImDrawData* draw_data = ImGui::GetDrawData();
    ImGui_ImplVulkan_RenderDrawData(draw_data, frame.command_buffer);

    vkCmdEndRenderingKHR(frame.command_buffer);

    barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask = 0;
    vkCmdPipelineBarrier(frame.command_buffer,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 
                         0, 0, 
                         0, 0, 
                         1, &barrier);

    vkr = vkEndCommandBuffer(frame.command_buffer);
    assert(vkr == VK_SUCCESS);

    vkr = gfx::Submit(frame, vk, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    assert(vkr == VK_SUCCESS);

    gfx::PresentResult present_result = gfx::PresentFrame(&window, &frame, vk);
    if (present_result != gfx::PresentResult::SUCCESS) {
        printf("Failed to submit\n");
        exit(1);
    }

    app->current_frame += 1;
}



int main(int argc, char** argv) {
    framegraph::test();
    exit(1);

    // Initialize glfw.
    glfwInit();

    // Check if device supports vulkan.
    if (!glfwVulkanSupported()) {
        printf("Vulkan not found!\n");
        exit(1);
    }

    // Get instance extensions required by GLFW.
    u32 glfw_instance_extensions_count;
    const char** glfw_instance_extensions = glfwGetRequiredInstanceExtensions(&glfw_instance_extensions_count);

    // Vulkan initialization.
    Array<const char*> instance_extensions;
    for (u32 i = 0; i < glfw_instance_extensions_count; i++) {
        instance_extensions.add(glfw_instance_extensions[i]);
    }
    instance_extensions.add("VK_EXT_debug_report");

    Array<const char*> device_extensions;
    device_extensions.add("VK_KHR_swapchain");
    device_extensions.add("VK_KHR_dynamic_rendering");

    u32 vulkan_api_version = VK_API_VERSION_1_2;

    gfx::Context vk = {};
    if (gfx::CreateContext(&vk, vulkan_api_version, instance_extensions, device_extensions, true, gfx::DeviceFeatures::DYNAMIC_RENDERING | gfx::DeviceFeatures::DESCRIPTOR_INDEXING, true, true) != gfx::Result::SUCCESS) {
        printf("Failed to initialize vulkan\n");
        exit(1);
    }

    // Check if queue family supports image presentation.
    // TODO: remove this once device picking logic is better
    if (!glfwGetPhysicalDevicePresentationSupport(vk.instance, vk.physical_device, vk.queue_family_index)) {
        printf("Device does not support image presentation\n");
        exit(1);
    }

    gfx::Window window = {};
    if (gfx::CreateWindowWithSwapchain(&window, vk, "XPG", 1600, 900, true) != gfx::Result::SUCCESS) {
        printf("Failed to create vulkan window\n");
        return 1;
    }

    gui::ImGuiImpl imgui_impl;
    gui::Create(&imgui_impl, window, vk);

    uint32_t MAX_DESCRIPTOR_COUNT = 100000;

    // BINDLES SLIM SETUP START

    gfx::BindlessDescriptorSet bindless = {};
    gfx::CreateBindlessDescriptorSet(&bindless, vk, {
        .entries = {
            {
                .count = 1000,
                .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            },
            {
                .count = 1000,
                .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            },
            {
                .count = 1000,
                .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            },
        },
    });

    // BINDLESS SLIM SETUP END


    // PIPELINE SETUP START
    VkResult vkr;

    //Array<u8> vertex_code = ReadEntireFile("res/bindless.vert.spirv");
    Array<u8> vertex_code = ReadEntireFile("res/bindless_slang.vert.spv");

    //Array<u8> fragment_code = ReadEntireFile("res/bindless.frag.spirv");
    Array<u8> fragment_code = ReadEntireFile("res/bindless_slang.frag.spv");

    gfx::Shader vertex_shader = {};
    vkr = gfx::CreateShader(&vertex_shader, vk, vertex_code);
    assert(vkr == VK_SUCCESS);

    gfx::Shader fragment_shader = {};
    vkr = gfx::CreateShader(&fragment_shader, vk, fragment_code);
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
                    .stride = sizeof(glm::vec3) + sizeof(glm::vec2),
                },
            },
            .vertex_attributes = {
                {
                    .location = 0,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                },
                {
                    .location = 1,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32_SFLOAT,
                    .offset = sizeof(glm::vec3),
                },
            },
            .depth = {
                .format = VK_FORMAT_D32_SFLOAT,
            },
            .push_constants = {
                {
                    .offset = 0,
                    .size = 4,
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
    assert(vkr == VK_SUCCESS);
    // PIPELINE SETUP END


    // Vertex data
    struct Vertex {
        vec3 pos;
        vec2 uv;
    };
    ArrayFixed<Vertex, 6> vertices(6);
    vertices[0] = { vec3(-0.5, -0.5, 0.0), vec2(0.0, 0.0) };
    vertices[1] = { vec3(0.5, -0.5, 0.0), vec2(1.0, 0.0) };
    vertices[2] = { vec3(0.5,  0.5, 0.0), vec2(1.0, 1.0) };
    vertices[3] = { vec3(0.5,  0.5, 0.0), vec2(1.0, 1.0) };
    vertices[4] = { vec3(-0.5,  0.5, 0.0), vec2(0.0, 1.0) };
    vertices[5] = { vec3(-0.5, -0.5, 0.0), vec2(0.0, 0.0) };
    size_t V = vertices.length;

    gfx::Buffer vertex_buffer = {};
    vkr = gfx::CreateBufferFromData(&vertex_buffer, vk, vertices.as_bytes(), {
            .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            .alloc_required_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            .alloc_preferred_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        });
    assert(vkr == VK_SUCCESS);


    // CREATE A BUFFER AND DESCRIPTOR START

    ArrayFixed<vec4, 4> colors(4);
    colors[0] = vec4(0, 0, 1, 1);
    colors[1] = vec4(1, 0, 0, 1);
    colors[2] = vec4(0, 1, 0, 1);
    colors[3] = vec4(1, 1, 1, 1);

    Array<gfx::Buffer> color_buffers(colors.length);
    for (size_t i = 0; i < colors.length; i++) {
        vkr = gfx::CreateBufferFromData(&color_buffers[i], vk, BytesOf(&colors[i]), {
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                .alloc_required_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                .alloc_preferred_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            });
        assert(vkr == VK_SUCCESS);

        // TODO: there should be an higher level construct
        // handling the bindless heap, keeping track of allocations
        // and offering helpers to alloc / write / bind / free descriptors.

        // TODO: also test doing the writes all together instead of 1 by 1

        // Prepare descriptor and handle
        uint32_t handle = (uint32_t)i;

        VkDescriptorBufferInfo buffer_info = {};
        buffer_info.buffer = color_buffers[i].buffer;
        buffer_info.offset = 0;
        buffer_info.range = VK_WHOLE_SIZE;

        VkWriteDescriptorSet write_descriptor_set = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        write_descriptor_set.dstSet = bindless.set;
        write_descriptor_set.dstArrayElement = handle;
        write_descriptor_set.descriptorCount = 1;
        write_descriptor_set.pBufferInfo = &buffer_info;
        write_descriptor_set.dstBinding = 0;       // Here we use 0 because in our descriptor bindings, we have STORAGE_BUFFER at index 0
        write_descriptor_set.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

        // Actually write the descriptor to the GPU visible heap
        vkUpdateDescriptorSets(vk.device, 1, &write_descriptor_set, 0, nullptr);
    }

    // CREATE A BUFFER AND DESCRIPTOR END

    App app = {};
    app.window = &window;
    app.vk = &vk;
    app.wait_for_events = true;
    app.last_frame_timestamp = glfwGetTimerValue();
    app.vertex_buffer = vertex_buffer;
    app.pipeline = pipeline.pipeline;
    app.layout = pipeline.layout;
    app.descriptor_set = bindless.set;

    glfwSetWindowUserPointer(window.window, &app);

    while (true) {
        if (app.wait_for_events) {
            glfwWaitEvents();
        }
        else {
            glfwPollEvents();
        }

        if (glfwWindowShouldClose(window.window)) {
            app.closed = true;
            break;
        }

        Draw(&app);
    }


    // Wait
    vkDeviceWaitIdle(vk.device);

    // Rendering stuff
    for (size_t i = 0; i < color_buffers.length; i++) {
        gfx::DestroyBuffer(&color_buffers[i], vk);
    }
    gfx::DestroyBuffer(&vertex_buffer, vk);
    gfx::DestroyShader(&vertex_shader, vk);
    gfx::DestroyShader(&fragment_shader, vk);
    gfx::DestroyGraphicsPipeline(&pipeline, vk);
    gfx::DestroyBindlessDescriptorSet(&bindless, vk);

    // Gui
    gui::Destroy(&imgui_impl, vk);
    
    // Window
    gfx::DestroyWindowWithSwapchain(&window, vk);

    // Context
    gfx::DestroyContext(&vk);

    //system("pause");
}
