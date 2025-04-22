#include <xpg/gui.h>
#include <xpg/log.h>
#include <xpg/platform.h>

using glm::vec2;
using glm::ivec2;
using glm::vec3;
using glm::vec4;
using glm::mat4;

int main(int argc, char** argv) {
    gfx::Result result;
    result = gfx::Init();
    if (result != gfx::Result::SUCCESS) {
        logging::error("bigimage", "Failed to initialize platform\n");
    }

    gfx::Context vk = {};
    result = gfx::CreateContext(&vk, {
        .minimum_api_version = (u32)VK_API_VERSION_1_1,
        .device_features = gfx::DeviceFeatures::PRESENTATION | gfx::DeviceFeatures::DYNAMIC_RENDERING | gfx::DeviceFeatures::DESCRIPTOR_INDEXING | gfx::DeviceFeatures::SYNCHRONIZATION_2,
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
    struct App {
        // - Window
        bool wait_for_events = true;
        bool closed = false;
        bool first_frame_done = false;

        // - UI
        platform::Timestamp last_frame_timestamp;
        gui::ImGuiImpl gui;
        Array<VkFramebuffer> framebuffers;

        // - Scene

        // - Rendering
        u32 frame_index = 0; // Rendering frame index, wraps around at the number of frames in flight
    };

    // USER: application
    App app = {};
    app.last_frame_timestamp = platform::GetTimestamp();
    app.framebuffers.resize(window.frames.length);

    auto MouseMoveEvent = [&app](ivec2 pos) {
    };

    auto MouseButtonEvent = [&app](ivec2 pos, gfx::MouseButton button, gfx::Action action, gfx::Modifiers mods) {
        if (ImGui::GetIO().WantCaptureMouse) return;
    };

    auto MouseScrollEvent = [&app](ivec2 pos, ivec2 scroll) {
        if (ImGui::GetIO().WantCaptureMouse) return;
    };

    auto Draw = [&app, &vk, &window]() {
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
            for(usize i = 0; i < app.framebuffers.length; i++) {
                vkDestroyFramebuffer(vk.device, app.framebuffers[i], 0);
                app.framebuffers[i] = VK_NULL_HANDLE;
            }
        }

        for(usize i = 0; i < app.framebuffers.length; i++) {
            if(app.framebuffers[i] != VK_NULL_HANDLE) continue;

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = app.gui.render_pass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = &window.image_views[i];
            framebufferInfo.width = window.fb_width;
            framebufferInfo.height = window.fb_height;
            framebufferInfo.layers = 1;

            VkResult vkr = vkCreateFramebuffer(vk.device, &framebufferInfo, nullptr, &app.framebuffers[i]);
            assert(vkr == VK_SUCCESS);
        }

        // Acquire current frame
        gfx::Frame& frame = gfx::WaitForFrame(&window, vk);
        gfx::Result ok = gfx::AcquireImage(&frame, &window, vk);
        if (ok != gfx::Result::SUCCESS) {
            return;
        }


        {
            // USER: pre-gui, but with frame

            {
                gui::BeginFrame();

                // USER: gui
                gui::DrawStats(dt, window.fb_width, window.fb_height);

                ImGui::ShowDemoWindow();

                gui::EndFrame();
            }

#if 0
            // USER: draw commands
            gfx::CmdImageBarrier(frame.command_buffer, {
                .image = frame.current_image,
                .src_stage = VK_PIPELINE_STAGE_2_NONE,
                .dst_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .src_access = 0,
                .dst_access = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
                .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            });

            VkClearColorValue color = { 0.1f, 0.1f, 0.1f, 1.0f };
            gfx::CmdBeginRendering(frame.command_buffer, {
                .color = {
                    {
                        .view = frame.current_image_view,
                        .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
                        .store_op = VK_ATTACHMENT_STORE_OP_STORE,
                        .clear = color,
                    },
                },
                .width = window.fb_width,
                .height = window.fb_height,
            });

            // Draw GUI
            gui::Render(frame.command_buffer);

            gfx::CmdEndRendering(frame.command_buffer);

            gfx::CmdImageBarrier(frame.command_buffer, {
                .image = frame.current_image,
                .src_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .dst_stage = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                .src_access = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                .dst_access = 0,
                .old_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .new_layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                });
#else

            // Record commands
            gfx::BeginCommands(frame.command_pool, frame.command_buffer, vk);

            VkClearColorValue color = { 0.1f, 0.2f, 0.4f, 1.0f };

            VkClearValue clear_values[1];
            clear_values[0].color = color;

            VkRenderPassBeginInfo info = {};
            info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            info.renderPass = app.gui.render_pass;
            info.framebuffer = app.framebuffers[frame.current_image_index];
            info.renderArea.extent.width = window.fb_width;
            info.renderArea.extent.height = window.fb_height;
            info.clearValueCount = ArrayCount(clear_values);
            info.pClearValues = clear_values;
            vkCmdBeginRenderPass(frame.command_buffer, &info, VK_SUBPASS_CONTENTS_INLINE);

            gui::Render(frame.command_buffer);

            vkCmdEndRenderPass(frame.command_buffer);

#endif

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

    gui::CreateImGuiImpl(&app.gui, window, vk, {
        .dynamic_rendering = false,
    });

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
    for(usize i = 0; i < app.framebuffers.length; i++) {
        vkDestroyFramebuffer(vk.device, app.framebuffers[i], 0);
    }

    // Gui
    gui::DestroyImGuiImpl(&app.gui, vk);

    // Window
    gfx::DestroyWindowWithSwapchain(&window, vk);

    // Context
    gfx::DestroyContext(&vk);
}
