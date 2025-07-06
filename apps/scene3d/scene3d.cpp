#include <xpg/gui.h>
#include <xpg/log.h>
#include <xpg/platform.h>

#include <imgui_internal.h>

using glm::vec2;
using glm::ivec2;
using glm::uvec2;
using glm::vec3;
using glm::vec4;
using glm::mat4;

using namespace xpg;

int main(int argc, char** argv) {
    gfx::Result result;
    result = gfx::Init();
    if (result != gfx::Result::SUCCESS) {
        logging::error("scene3d", "Failed to initialize platform\n");
    }

    gfx::Context vk = {};
    result = gfx::CreateContext(&vk, {
        .minimum_api_version = (u32)VK_API_VERSION_1_1,
        .required_features = gfx::DeviceFeatures::DYNAMIC_RENDERING | gfx::DeviceFeatures::DESCRIPTOR_INDEXING | gfx::DeviceFeatures::SYNCHRONIZATION_2,
        .enable_validation_layer = true,
        //        .enable_gpu_based_validation = true,
    });

    if (result != gfx::Result::SUCCESS) {
        logging::error("scene3d", "Failed to initialize vulkan\n");
        exit(100);
    }

    gfx::Window window = {};
    result = gfx::CreateWindowWithSwapchain(&window, vk, "XPG", 1600, 900);
    if (result != gfx::Result::SUCCESS) {
        logging::error("scene3d", "Failed to create vulkan window\n");
        exit(100);
    }

    gui::ImGuiImpl gui;
    gui::CreateImGuiImpl(&gui, window, vk, {});

    // gui::SetDarkTheme();
    ImGuiStyle& style = ImGui::GetStyle();
    style.TabRounding = 0.0f;
    style.GrabRounding = 0.0f;
    style.ChildRounding = 0.0f;
    style.FrameRounding = 0.0f;
    style.PopupRounding = 0.0f;
    style.WindowRounding = 0.0f;

    VkResult vkr;

    // Descriptors
    struct App {
        // - Window
        bool wait_for_events = true;
        bool closed = false;
        bool first_frame_done = false;

        // - UI
        platform::Timestamp last_frame_timestamp;

        // - Scene

        // - Rendering
        u32 frame_index = 0; // Rendering frame index, wraps around at the number of frames in flight
    };

    // USER: application
    App app = {};
    app.last_frame_timestamp = platform::GetTimestamp();

    auto MouseMoveEvent = [&app](ivec2 pos) {
    };

    auto MouseButtonEvent = [&app](ivec2 pos, gfx::MouseButton button, gfx::Action action, gfx::Modifiers mods) {
        if (ImGui::GetIO().WantCaptureMouse) return;
    };

    auto MouseScrollEvent = [&app](ivec2 pos, ivec2 scroll) {
        if (ImGui::GetIO().WantCaptureMouse) return;
    };

    struct Viewport {
        uvec2 pos;
        uvec2 size;
        VkClearColorValue color;
        VkCommandBuffer cmd;

        static void draw(const ImDrawList* draw_list, const ImDrawCmd* draw_cmd) {
            Viewport* v = (Viewport*)draw_cmd->UserCallbackData;

            VkCommandBuffer cmd  = *(VkCommandBuffer*)ImGui::GetPlatformIO().Renderer_RenderState;

            VkClearAttachment attachment;
            attachment.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            attachment.colorAttachment = 0;
            attachment.clearValue.color = v->color;

            VkClearRect rect;
            rect.baseArrayLayer = 0;
            rect.layerCount = 1;
            rect.rect.offset.x = v->pos.x;
            rect.rect.offset.y = v->pos.y;
            rect.rect.extent.width =  v->size.x;
            rect.rect.extent.height = v->size.y;
            vkCmdClearAttachments(cmd, 1, &attachment, 1, &rect);
        }
    };

    ArrayFixed<Viewport, 3> viewports(3);
    viewports[0].color = { .float32 = { 1, 0, 0, 1 } };
    viewports[1].color = { .float32 = { 0, 1, 0, 1 } };
    viewports[2].color = { .float32 = { 0, 0, 1, 1 } };

    auto Draw = [&app, &vk, &window, &viewports]() {
        if (app.closed) return;

        platform::Timestamp timestamp = platform::GetTimestamp();
        float dt = (float)platform::GetElapsed(app.last_frame_timestamp, timestamp);
        app.last_frame_timestamp = timestamp;

        gfx::SwapchainStatus swapchain_status = gfx::UpdateSwapchain(&window, vk);
        if (swapchain_status == gfx::SwapchainStatus::FAILED) {
            logging::error("scene3d/draw", "Swapchain update failed\n");
            exit(101);
        }
        else if (swapchain_status == gfx::SwapchainStatus::MINIMIZED) {
            return;
        }

        if (swapchain_status == gfx::SwapchainStatus::RESIZED || !app.first_frame_done) {
            app.first_frame_done = true;

            // USER: resize (e.g. framebuffer sized elements)
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
                // gui::DrawStats(dt, window.fb_width, window.fb_height);

                ImGui::DockSpaceOverViewport(0, 0, ImGuiDockNodeFlags_PassthruCentralNode);

                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));

                for (usize i = 0; i < viewports.length; i++) {
                    char window_name[1024];
                    snprintf(window_name, sizeof(window_name), "Hello %d", (int)i);

                    ImGuiWindowClass window_class;
                    window_class.DockNodeFlagsOverrideSet = ImGuiDockNodeFlags_AutoHideTabBar;
                    ImGui::SetNextWindowClass(&window_class);

                    if (ImGui::Begin(window_name)) {
                        ImVec2 pos = ImGui::GetCursorScreenPos();
                        ImVec2 size = ImGui::GetContentRegionAvail();

                        viewports[i].pos = uvec2(pos.x, pos.y);
                        viewports[i].size = uvec2(size.x, size.y);

                        ImGuiWindow* w = ImGui::GetCurrentWindow();
                        printf("window %zu: %d\n", i, w->DockIsActive);

                        ImDrawList* list = ImGui::GetWindowDrawList();
                        bool channel = false;
                        if (w->DockIsActive && w->DockNode->IsHiddenTabBar()) {
                            list = w->DockNode->HostWindow->DrawList;
                            list->ChannelsSetCurrent(DOCKING_HOST_DRAW_CHANNEL_BG);
                            channel = true;
                        }
                        list->AddCallback(Viewport::draw, &viewports[i]);
                        if (channel) {
                            list->ChannelsSetCurrent(DOCKING_HOST_DRAW_CHANNEL_FG);
                        }
                    }
                    ImGui::End();
                }
                ImGui::PopStyleVar();
                gui::EndFrame();
            }

            // USER: draw commands
            gfx::BeginCommands(frame.command_pool, frame.command_buffer, vk);

            gfx::CmdImageBarrier(frame.command_buffer, {
                .src_stage = VK_PIPELINE_STAGE_2_NONE,
                .src_access = 0,
                .dst_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .dst_access = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
                .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .image = frame.current_image,
            });

            VkClearColorValue color = { 0.3f, 0.1f, 0.1f, 1.0f };
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

    while (true) {
        gfx::ProcessEvents(app.wait_for_events);

        if (gfx::ShouldClose(window)) {
            logging::info("scene3d", "Window closed");
            app.closed = true;
            break;
        }

        // Draw
        Draw();
    };

    // Wait
    gfx::WaitIdle(vk);

    // USER: cleanup

    // Gui
    gui::DestroyImGuiImpl(&gui, vk);

    // Window
    gfx::DestroyWindowWithSwapchain(&window, vk);

    // Context
    gfx::DestroyContext(&vk);
}
