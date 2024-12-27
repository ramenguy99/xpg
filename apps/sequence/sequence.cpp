#include <xpg/gui.h>
#include <xpg/buffered_stream.h>

#include <math.h>

using glm::vec2;
using glm::ivec2;
using glm::vec3;
using glm::vec4;
using glm::mat4;

#include "types.h"

struct App {
    gfx::Context* vk;
    gfx::Window* window;

    VkQueue queue;

    // Swapchain frames, index wraps around at the number of frames in flight.
    u32 frame_index;
    Array<gfx::Frame> frames;
    // Total frame index.
    u64 current_frame;

    bool force_swapchain_update;
    bool wait_for_events;
    bool closed;

    // Application data.
    u64 last_frame_timestamp;
    ArrayFixed<f32, 64> frame_times;
    Array<VkDescriptorSet> descriptor_sets;
    Array<gfx::Buffer> uniform_buffers;
    gfx::DepthBuffer depth_buffer;

    // Playback
    bool playback_enabled;
    u32 num_playback_frames;
    u32 playback_frame;
    f64 playback_delta;

    BufferedStream<platform::FileReadWork> mesh_stream;
    ArrayView<u8> vertex_map;
    platform::File mesh_file;
    u32 num_vertices;
    u32 num_indices;

    // Rendering
    VkBuffer vertex_buffer;
    VkBuffer index_buffer;
    VkPipeline pipeline;
    VkPipelineLayout layout;
};

void Draw(App* app) {
    if (app->closed) return;

    auto& vk = *app->vk;
    auto& window = *app->window;

    u64 timestamp = glfwGetTimerValue();

    float dt = (float)((double)(timestamp - app->last_frame_timestamp) / (double)glfwGetTimerFrequency());
    app->last_frame_timestamp = timestamp;
    if (isnan(dt) || isinf(dt)) {
        dt = 0.0f;
    }
    app->frame_times[app->current_frame % app->frame_times.length] = dt;

    float avg_frame_time = 0.0f;
    for (usize i = 0; i < app->frame_times.length; i++) {
        avg_frame_time += app->frame_times[i];
    }
    avg_frame_time /= (f32)app->frame_times.length;

    gfx::SwapchainStatus swapchain_status = gfx::UpdateSwapchain(&window, vk);
    if (swapchain_status == gfx::SwapchainStatus::FAILED) {
        printf("Swapchain update failed\n");
        exit(1);
    }
    app->force_swapchain_update = false;

    if (swapchain_status == gfx::SwapchainStatus::MINIMIZED) {
        app->wait_for_events = true;
        return;
    }
    else if(swapchain_status == gfx::SwapchainStatus::RESIZED) {
        // Resize framebuffer sized elements.
        gfx::DestroyDepthBuffer(&app->depth_buffer, vk);
        VkResult vkr = gfx::CreateDepthBuffer(&app->depth_buffer, vk, window.fb_width, window.fb_height);
        if(vkr != VK_SUCCESS) {
            printf("Depth buffer resize failed\n");
            exit(1);
        }
    }

    app->wait_for_events = false;

    // Acquire current frame
    gfx::Frame& frame = app->frames[app->frame_index];

    vkWaitForFences(vk.device, 1, &frame.fence, VK_TRUE, ~0);

    u32 index;
    VkResult vkr = vkAcquireNextImageKHR(vk.device, window.swapchain, ~0ull, frame.acquire_semaphore, 0, &index);
    if(vkr == VK_ERROR_OUT_OF_DATE_KHR) {
        app->force_swapchain_update = true;
        return;
    }

    // Playback update
    if (app->playback_enabled) {
        app->playback_delta += dt;
        u32 frames_to_step = (u32)(app->playback_delta * 25.0f);
        app->playback_frame = (app->playback_frame + frames_to_step + app->num_playback_frames) % app->num_playback_frames;
        app->playback_delta -= frames_to_step * (1.0f / 25.0f);
    }


    // ImGui
    gui::BeginFrame();

    ImGui::DockSpaceOverViewport(NULL, ImGuiDockNodeFlags_PassthruCentralNode);

    if (ImGui::Begin("Playback")) {
        struct Getter {
            static float fn(void* data, int index) {
                App* a = (App*)data;

                usize i = (index - (a->current_frame % a->frame_times.length) + a->frame_times.length) % a->frame_times.length;
                return 1.0f / a->frame_times[a->frame_times.length - i - 1];
            }
        };
        //ImGui::PlotLines("", app->frame_times.data, (int)app->frame_times.length, 0, 0, 0.0f, .0f, ImVec2(100, 30));
        ImGui::PlotLines("", Getter::fn, app, (int)app->frame_times.length, 0, 0, 0.0f, 200.0f, ImVec2(100, 30));
        ImGui::SameLine();
        ImGui::Text("FPS: %.2f (%.2fms) [%.2f (%.2fms)]", 1.0f / dt, dt * 1.0e3f, 1.0 / avg_frame_time, avg_frame_time * 1.0e3f);

        int frame = app->playback_frame;
        if (ImGui::SliderInt("Playback", &frame, 0, app->num_playback_frames - 1)) {
            app->playback_frame = frame;
            app->playback_delta = 0.0f;
        }

        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 p = ImGui::GetCursorScreenPos();
        f32 x = p.x + 2.0f;
        f32 y = p.y + 4.0f;
        f32 width = 4.0f;
        f32 height = 16.0f;
        for (u32 i = 0; i < app->num_playback_frames; i++) {
            // Normalize frame index
            u64 frame_index = i;
            if (frame_index < app->mesh_stream.stream_cursor) {
                frame_index += app->mesh_stream.stream_length;
            }

            // If in bounds of the buffer
            if (frame_index < app->mesh_stream.stream_cursor + app->mesh_stream.buffer.length) {
                u64 delta = frame_index - app->mesh_stream.stream_cursor;
                u64 buffer_index = (app->mesh_stream.buffer_offset + delta) % app->mesh_stream.buffer.length;

                u32 color = 0xFF000000;

                using platform::FileReadWork;
                BufferedStream<FileReadWork>::EntryState state = app->mesh_stream.buffer[buffer_index].state.load(std::memory_order_relaxed);
                switch (state) {
                case BufferedStream<FileReadWork>::EntryState::Empty: color = 0xFF000000; break;
                case BufferedStream<FileReadWork>::EntryState::Filling: color = 0xFF00FFFF; break;
                case BufferedStream<FileReadWork>::EntryState::Canceling: color = 0xFF0000FF; break;
                case BufferedStream<FileReadWork>::EntryState::Done: color = 0xFF00FF00; break;
                }
                draw_list->AddRectFilled(ImVec2(x, y), ImVec2(x + width, y + height), color);
            }

            u32 outline_color = 0xFFFFFFFF;
            if (app->mesh_stream.stream_cursor == i) {
                outline_color = 0xFF000000;
            }
            draw_list->AddRect(ImVec2(x, y), ImVec2(x + width, y + height), outline_color, 0, 0, 1.0f);


            x += width + 1.0f;
        }
    }
    ImGui::End();
    ImGui::ShowDemoWindow();

    // Render imgui.
    ImGui::Render();

    // Reset command pool
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
    barrier.image = window.images[index];
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

    {
        VkImageMemoryBarrier barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = app->depth_buffer.image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;

        barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;

        vkCmdPipelineBarrier(frame.command_buffer,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0,
                             0, 0,
                             0, 0,
                             1, &barrier);
    }

    // Begin rendering.
    VkRenderingAttachmentInfo attachment_info = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    attachment_info.imageView = window.image_views[index];
    attachment_info.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachment_info.resolveMode = VK_RESOLVE_MODE_NONE;
    attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment_info.clearValue.color = color;

    VkRenderingAttachmentInfo depth_attachment_info = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    depth_attachment_info.imageView = app->depth_buffer.view;
    depth_attachment_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth_attachment_info.resolveMode = VK_RESOLVE_MODE_NONE;
    depth_attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_attachment_info.clearValue.depthStencil.depth = 1.0;

    VkRenderingInfo rendering_info = { VK_STRUCTURE_TYPE_RENDERING_INFO };
    rendering_info.renderArea.extent.width = window.fb_width;
    rendering_info.renderArea.extent.height = window.fb_height;
    rendering_info.layerCount = 1;
    rendering_info.viewMask = 0;
    rendering_info.colorAttachmentCount = 1;
    rendering_info.pColorAttachments = &attachment_info;
    rendering_info.pDepthAttachment = &depth_attachment_info;

    vkCmdBeginRenderingKHR(frame.command_buffer, &rendering_info);

    vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app->pipeline);


    //u32 animation_frame = app->current_frame % app->num_frames;
    u32 size = app->num_vertices * sizeof(vec3);
    u32 buffer_offset = size * app->frame_index;

#if 0
    Array<vec3> vertices(app->num_vertices);
    ReadAtOffset(app->mesh_file, vertices.as_bytes(), 12 + (u64)size * animation_frame);
    memcpy(app->vertex_map.data + buffer_offset, vertices.data, size);
#else
    platform::FileReadWork w = app->mesh_stream.get_frame(app->playback_frame);
    memcpy(app->vertex_map.data + buffer_offset, w.buffer.data, size);
#endif

    VkDeviceSize offsets[1] = { buffer_offset };
    vkCmdBindVertexBuffers(frame.command_buffer, 0, 1, &app->vertex_buffer, offsets);

    vkCmdBindIndexBuffer(frame.command_buffer, app->index_buffer, 0, VK_INDEX_TYPE_UINT32);

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
    f32 ar = (f32)app->window->fb_width / (f32)app->window->fb_height;

    Constants* constants = app->uniform_buffers[app->frame_index].map.as_type<Constants>();
    constants->color = vec3(0.8, 0.8, 0.8);
    constants->transform = glm::perspective(fov, ar, 0.01f, 100.0f) * glm::lookAt(camera_position, camera_target, vec3(0, 1, 0));

    vkCmdBindDescriptorSets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app->layout, 0, 1, &app->descriptor_sets[app->frame_index], 0, 0);

    vkCmdDrawIndexed(frame.command_buffer, app->num_indices, 1, 0, 0, 0);

    vkCmdEndRenderingKHR(frame.command_buffer);

    attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    rendering_info.pDepthAttachment = 0;
    vkCmdBeginRenderingKHR(frame.command_buffer, &rendering_info);

    gui::Render(frame.command_buffer);

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

    // Submit commands
    VkPipelineStageFlags submit_stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSubmitInfo submit_info = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &frame.command_buffer;
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = &frame.acquire_semaphore;
    submit_info.pWaitDstStageMask = &submit_stage_mask;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &frame.release_semaphore;


    vkr = vkQueueSubmit(app->queue, 1, &submit_info, frame.fence);
    assert(vkr == VK_SUCCESS);

    // Present
    VkPresentInfoKHR present_info = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &window.swapchain;
    present_info.pImageIndices = &index;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = &frame.release_semaphore;
    vkr = vkQueuePresentKHR(app->queue, &present_info);

    if (vkr == VK_ERROR_OUT_OF_DATE_KHR || vkr == VK_SUBOPTIMAL_KHR) {
        app->force_swapchain_update = true;
    } else if (vkr != VK_SUCCESS) {
        printf("Failed to submit\n");
        exit(1);
    }

    // // Wait
    // vkr = vkDeviceWaitIdle(vk.device);
    // assert(vkr == VK_SUCCESS);
    app->frame_index = (app->frame_index + 1) % window.images.length;
    app->current_frame += 1;
}

static void
Callback_Key(GLFWwindow* window, int key, int scancode, int action, int mods) {
    App* app = (App*)glfwGetWindowUserPointer(window);

    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        if (key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(app->window->window, true);
        }

        if (key == GLFW_KEY_PERIOD) {
            app->playback_frame += 1;
            app->playback_frame = (app->playback_frame + app->num_playback_frames) % app->num_playback_frames;
        }

        if (key == GLFW_KEY_COMMA) {
            app->playback_frame -= 1;
            app->playback_frame = (app->playback_frame + app->num_playback_frames) % app->num_playback_frames;
        }

        if (key == GLFW_KEY_SPACE) {
            app->playback_enabled = !app->playback_enabled;
            app->playback_delta = 0.0f;
        }
    }
}

static void
Callback_WindowResize(GLFWwindow* window, int width, int height) {
    App* app = (App*)glfwGetWindowUserPointer(window);
}

static void
Callback_WindowRefresh(GLFWwindow* window) {
    App* app = (App*)glfwGetWindowUserPointer(window);
    if (app) {
        Draw(app);
    }
}

#ifdef _WIN32
DWORD WINAPI thread_proc(void* param) {
    HWND window = (HWND)param;
    while (true) {
        SendMessage(window, WM_PAINT, 0, 0);
    }
    return 0;
}
#endif

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

    u64 size = V * sizeof(vec3);

    u32 WORKERS = 4;
    u32 BUFFER_SIZE = (u32)8;

    Array<u8> loaded_data(size * BUFFER_SIZE);
    ArrayView<u8> loaded_data_view = loaded_data;

    WorkerPool pool;
    pool.init(WORKERS);

    using platform::FileReadWork;
    BufferedStream<FileReadWork> mesh_stream(N, BUFFER_SIZE, &pool,
        // Init
        [=](u64 index, u64 buffer_index, bool high_priority) mutable {
            FileReadWork w = {};

            w.file = file;
            w.offset = 12 + size * index;
            w.buffer = loaded_data_view.slice(buffer_index * size, size);
            w.do_chunks = true;

            return w;
    },
        // Fill
        [=](FileReadWork* w) {
            u64 bytes_left = w->buffer.length - w->bytes_read;

            u64 chunk_size = w->do_chunks ? Min((u64)16 * 1024, bytes_left) : bytes_left;
            ReadAtOffset(w->file, w->buffer.slice(w->bytes_read, chunk_size), w->offset + w->bytes_read);
            w->bytes_read += chunk_size;

            //printf("Read data: %llu  %llu\n", w->buffer.length, w->offset);
            return w->bytes_read == w->buffer.length;
        }
    );

    gfx::Result result;
    result = gfx::Init();
    if (result != gfx::Result::SUCCESS) {
        logging::error("sequence", "Failed to initialize platform\n");
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
        logging::error("sequence", "Failed to initialize vulkan\n");
        exit(100);
    }
    gfx::Window window = {};
    if (gfx::CreateWindowWithSwapchain(&window, vk, "XPG", 1600, 900) != gfx::Result::SUCCESS) {
        printf("Failed to create vulkan window\n");
        return 1;
    }

    // Initialize queue and command allocator.
    VkResult vkr;
    VkQueue queue;
    vkGetDeviceQueue(vk.device, vk.queue_family_index, 0, &queue);

    Array<gfx::Frame> frames(window.images.length);
    for (usize i = 0; i < frames.length; i++) {
        gfx::Frame& frame = frames[i];

        VkCommandPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;// | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = vk.queue_family_index;

        vkr = vkCreateCommandPool(vk.device, &pool_info, 0, &frame.command_pool);
        assert(vkr == VK_SUCCESS);

        VkCommandBufferAllocateInfo allocate_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        allocate_info.commandPool = frame.command_pool;
        allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate_info.commandBufferCount = 1;

        vkr = vkAllocateCommandBuffers(vk.device, &allocate_info, &frame.command_buffer);
        assert(vkr == VK_SUCCESS);

        VkFenceCreateInfo fence_info = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(vk.device, &fence_info, 0, &frame.fence);

        gfx::CreateGPUSemaphore(vk.device, &frame.acquire_semaphore);
        gfx::CreateGPUSemaphore(vk.device, &frame.release_semaphore);
    }

    // Setup window callbacks
    glfwSetWindowSizeCallback(window.window, Callback_WindowResize);
    glfwSetWindowRefreshCallback(window.window, Callback_WindowRefresh);
    glfwSetKeyCallback(window.window, Callback_Key);

    gui::ImGuiImpl imgui_impl;
    gui::CreateImGuiImpl(&imgui_impl, window, vk, {});

    VkBufferCreateInfo vertex_buffer_info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    vertex_buffer_info.size = sizeof(vec3) * V * window.images.length;
    vertex_buffer_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

    VkBufferCreateInfo index_buffer_info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    index_buffer_info.size = indices.size_in_bytes();
    index_buffer_info.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

    VmaAllocationCreateInfo alloc_create_info = {};
    alloc_create_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    alloc_create_info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    alloc_create_info.flags = 0;
        // VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT  <- Host can access, useful if usage=AUTO, not necessary if require HOST_VISIBLE
        // VMA_ALLOCATION_CREATE_MAPPED_BIT              <- Persistently mapped, not needed for upload once usage as we are doing here, we will manually map and unmap.

    VkBuffer vertex_buffer = 0;
    VmaAllocation vertex_allocation = {};
    alloc_create_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo vertex_alloc_info = {};
    vkr = vmaCreateBuffer(vk.vma, &vertex_buffer_info, &alloc_create_info, &vertex_buffer, &vertex_allocation, &vertex_alloc_info);
    assert(vkr == VK_SUCCESS);
    ArrayView<u8> vertex_map = ArrayView<u8>((u8*)vertex_alloc_info.pMappedData, vertex_buffer_info.size);

    VkBuffer index_buffer = 0;
    VmaAllocation index_allocation = {};
    alloc_create_info.flags = 0;
    vkr = vmaCreateBuffer(vk.vma, &index_buffer_info, &alloc_create_info, &index_buffer, &index_allocation, 0);
    assert(vkr == VK_SUCCESS);


    void* map;
    vmaMapMemory(vk.vma, index_allocation, &map);
    ArrayView<u8> index_map((u8*)map, index_buffer_info.size);
    index_map.copy_exact(indices.as_bytes());
    vmaUnmapMemory(vk.vma, index_allocation);


    // Create graphics pipeline.
    Array<u8> vertex_code;
    if (platform::ReadEntireFile("res/basic.vert.spirv", &vertex_code) != platform::Result::Success) {
        logging::error("sequence", "Failed to read vertex shader\n");
        exit(100);
    }
    Array<u8> fragment_code;
    if (platform::ReadEntireFile("res/basic.frag.spirv", &fragment_code) != platform::Result::Success) {
        logging::error("sequence", "Failed to read fragment shader\n");
        exit(100);
    }

    VkShaderModuleCreateInfo vertex_module_info = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    vertex_module_info.codeSize = vertex_code.length;
    vertex_module_info.pCode = (u32*)vertex_code.data;
    VkShaderModule vertex_module = 0;
    vkr = vkCreateShaderModule(vk.device, &vertex_module_info, 0, &vertex_module);
    assert(vkr == VK_SUCCESS);

    VkShaderModuleCreateInfo fragment_module_info = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    fragment_module_info.codeSize = fragment_code.length;
    fragment_module_info.pCode = (u32*)fragment_code.data;
    VkShaderModule fragment_module = 0;
    vkCreateShaderModule(vk.device, &fragment_module_info, 0, &fragment_module);
    assert(vkr == VK_SUCCESS);

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].flags = 0;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertex_module;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].flags = 0;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragment_module;
    stages[1].pName = "main";

    VkVertexInputBindingDescription vertex_bindings[1] = {};
    vertex_bindings[0].binding = 0;
    vertex_bindings[0].stride = sizeof(glm::vec3);
    vertex_bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription vertex_attributes[1] = {};
    vertex_attributes[0].location = 0;
    vertex_attributes[0].binding = 0;
    vertex_attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    vertex_attributes[0].offset = 0;

    VkPipelineVertexInputStateCreateInfo vertex_input_state = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    vertex_input_state.vertexBindingDescriptionCount = ArrayCount(vertex_bindings);
    vertex_input_state.pVertexBindingDescriptions = vertex_bindings;
    vertex_input_state.vertexAttributeDescriptionCount = ArrayCount(vertex_attributes);
    vertex_input_state.pVertexAttributeDescriptions = vertex_attributes;

    VkPipelineInputAssemblyStateCreateInfo input_assembly_state = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    input_assembly_state.primitiveRestartEnable = false;
    input_assembly_state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineTessellationStateCreateInfo tessellation_state = { VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO };

    VkPipelineViewportStateCreateInfo viewport_state = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterization_state = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rasterization_state.depthClampEnable = false;
    rasterization_state.rasterizerDiscardEnable = false;
    rasterization_state.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization_state.cullMode = VK_CULL_MODE_NONE;
    rasterization_state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization_state.depthBiasEnable = false;
    rasterization_state.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample_state = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    multisample_state.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth_stencil_state = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    depth_stencil_state.depthTestEnable = true;
    depth_stencil_state.depthWriteEnable = true;
    depth_stencil_state.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState attachments[1] = {};
    attachments[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    // attachments[0].blendEnable = VK_TRUE;
    // attachments[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    // attachments[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    // attachments[0].colorBlendOp = VK_BLEND_OP_ADD;
    // attachments[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    // attachments[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    // attachments[0].alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo blend_state = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    blend_state.attachmentCount = ArrayCount(attachments);
    blend_state.pAttachments = attachments;

    VkDynamicState dynamic_states[2] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    VkPipelineDynamicStateCreateInfo dynamic_state = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynamic_state.dynamicStateCount = ArrayCount(dynamic_states);
    dynamic_state.pDynamicStates = dynamic_states;

    // Rendering
    VkFormat color_formats[1] = {
        window.swapchain_format
    };

    VkPipelineRenderingCreateInfo rendering_create_info = { VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    rendering_create_info.colorAttachmentCount = 1;
    rendering_create_info.pColorAttachmentFormats = color_formats;
    rendering_create_info.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

    // Layout
    VkDescriptorSetLayoutBinding binding = {};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_ALL;

    VkDescriptorSetLayoutCreateInfo descriptor_set_layout_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    descriptor_set_layout_info.bindingCount = 1;
    descriptor_set_layout_info.pBindings = &binding;

    VkDescriptorSetLayout descriptor_set_layout = 0;
    vkr = vkCreateDescriptorSetLayout(vk.device, &descriptor_set_layout_info, 0, &descriptor_set_layout);
    assert(vkr == VK_SUCCESS);

    VkPipelineLayoutCreateInfo layout_info = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &descriptor_set_layout;

    VkPipelineLayout layout = 0;
    vkr = vkCreatePipelineLayout(vk.device, &layout_info, 0, &layout);
    assert(vkr == VK_SUCCESS);

    VkGraphicsPipelineCreateInfo pipeline_create_info = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pipeline_create_info.pNext = &rendering_create_info;
    pipeline_create_info.flags = 0;

    // Shaders
    pipeline_create_info.stageCount = 2;
    pipeline_create_info.pStages = stages;

    // Graphics state
    pipeline_create_info.pVertexInputState = &vertex_input_state;
    pipeline_create_info.pInputAssemblyState = &input_assembly_state;
    pipeline_create_info.pTessellationState = &tessellation_state;
    pipeline_create_info.pViewportState = &viewport_state;
    pipeline_create_info.pRasterizationState = &rasterization_state;
    pipeline_create_info.pMultisampleState = &multisample_state;
    pipeline_create_info.pDepthStencilState = &depth_stencil_state;
    pipeline_create_info.pColorBlendState = &blend_state;
    pipeline_create_info.pDynamicState = &dynamic_state;

    // Binding layout
    pipeline_create_info.layout = layout;

    // Render pass -> we use dynamic rendering
    pipeline_create_info.renderPass = VK_NULL_HANDLE;

    VkPipeline pipeline = 0;
    vkr = vkCreateGraphicsPipelines(vk.device, VK_NULL_HANDLE, 1, &pipeline_create_info, 0, &pipeline);
    assert(vkr == VK_SUCCESS);

    // Create a descriptor set for shader constnats.
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, (u32)frames.length},
    };

    VkDescriptorPoolCreateInfo descriptor_pool_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    descriptor_pool_info.flags = 0;
    descriptor_pool_info.maxSets = (u32)frames.length;
    descriptor_pool_info.pPoolSizes = pool_sizes;
    descriptor_pool_info.poolSizeCount = ArrayCount(pool_sizes);

    VkDescriptorPool descriptor_pool = 0;
    vkCreateDescriptorPool(vk.device, &descriptor_pool_info, 0, &descriptor_pool);

    Array<VkDescriptorSetLayout> descriptor_set_layouts(frames.length);
    for (usize i = 0; i < descriptor_set_layouts.length; i++) {
        descriptor_set_layouts[i] = descriptor_set_layout;
    }

    VkDescriptorSetAllocateInfo descriptor_set_alloc_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    descriptor_set_alloc_info.descriptorPool = descriptor_pool;
    descriptor_set_alloc_info.descriptorSetCount = (u32)descriptor_set_layouts.length;
    descriptor_set_alloc_info.pSetLayouts = descriptor_set_layouts.data;

    Array<VkDescriptorSet> descriptor_sets(frames.length);
    vkr = vkAllocateDescriptorSets(vk.device, &descriptor_set_alloc_info, descriptor_sets.data);
    assert(vkr == VK_SUCCESS);

    alloc_create_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    Array<gfx::Buffer> uniform_buffers(frames.length);
    for (usize i = 0; i < uniform_buffers.length; i++) {
        VkBufferCreateInfo buffer_info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        buffer_info.size = sizeof(Constants);
        buffer_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;


        VmaAllocationInfo alloc_info = {};
        vkr = vmaCreateBuffer(vk.vma, &buffer_info, &alloc_create_info, &uniform_buffers[i].buffer, &uniform_buffers[i].allocation, &alloc_info);
        assert(vkr == VK_SUCCESS);

        uniform_buffers[i].map = ArrayView<u8>((u8*)alloc_info.pMappedData, buffer_info.size);

        VkDescriptorBufferInfo buffer_descriptor_info = {};
        buffer_descriptor_info.buffer = uniform_buffers[i].buffer;
        buffer_descriptor_info.offset = 0;
        buffer_descriptor_info.range = VK_WHOLE_SIZE;

        VkWriteDescriptorSet descriptor_write = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        descriptor_write.dstSet = descriptor_sets[i];
        descriptor_write.dstBinding = 0;
        descriptor_write.dstArrayElement = 0;
        descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptor_write.descriptorCount = 1;
        descriptor_write.pBufferInfo = &buffer_descriptor_info;
        vkUpdateDescriptorSets(vk.device, 1, &descriptor_write, 0, 0);
    }

    gfx::DepthBuffer depth_buffer;
    vkr = gfx::CreateDepthBuffer(&depth_buffer, vk, window.fb_width, window.fb_height);
    assert(vkr == VK_SUCCESS);

    App app = {};
    app.frames = std::move(frames);
    app.window = &window;
    app.vk = &vk;
    app.queue = queue;
    app.wait_for_events = true;
    app.frame_times.resize(ArrayCount(app.frame_times.data));
    app.last_frame_timestamp = glfwGetTimerValue();
    app.vertex_buffer = vertex_buffer;
    app.index_buffer = index_buffer;
    app.pipeline = pipeline;
    app.descriptor_sets = std::move(descriptor_sets);
    app.uniform_buffers = std::move(uniform_buffers);
    app.layout = layout;
    app.depth_buffer = depth_buffer;

    app.num_playback_frames = (u32)N;
    app.num_vertices = (u32)V;
    app.num_indices = (u32)I;
    app.vertex_map = vertex_map;
    app.mesh_file = file;
    app.mesh_stream = std::move(mesh_stream);

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
    gfx::WaitIdle(vk);

    pool.destroy();

    gfx::DestroyDepthBuffer(&app.depth_buffer, vk);

    vkDestroyDescriptorPool(vk.device, descriptor_pool, 0);
    vkDestroyDescriptorSetLayout(vk.device, descriptor_set_layout, 0);

    for (usize i = 0; i < app.uniform_buffers.length; i++) {
        vmaDestroyBuffer(vk.vma, app.uniform_buffers[i].buffer, app.uniform_buffers[i].allocation);
    }

    vmaDestroyBuffer(vk.vma, vertex_buffer, vertex_allocation);
    vmaDestroyBuffer(vk.vma, index_buffer, index_allocation);

    vkDestroyShaderModule(vk.device, vertex_module, 0);
    vkDestroyShaderModule(vk.device, fragment_module, 0);
    vkDestroyPipelineLayout(vk.device, layout, 0);
    vkDestroyPipeline(vk.device, pipeline, 0);

    for (usize i = 0; i < window.image_views.length; i++) {
        gfx::Frame& frame = app.frames[i];
        vkDestroyFence(vk.device, frame.fence, 0);

        vkDestroySemaphore(vk.device, frame.acquire_semaphore, 0);
        vkDestroySemaphore(vk.device, frame.release_semaphore, 0);

        vkFreeCommandBuffers(vk.device, frame.command_pool, 1, &frame.command_buffer);
        vkDestroyCommandPool(vk.device, frame.command_pool, 0);
    }

    // Gui
    gui::DestroyImGuiImpl(&imgui_impl, vk);

    // Window
    gfx::DestroyWindowWithSwapchain(&window, vk);

    // Context
    gfx::DestroyContext(&vk);
}
