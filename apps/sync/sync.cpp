#include <xpg/gui.h>
#include <xpg/log.h>

using namespace xpg;

int main(int argc, char** argv) {
    gfx::Result result;
    result = gfx::Init();
    if (result != gfx::Result::SUCCESS) {
        logging::error("sync", "Failed to initialize platform\n");
        exit(100);
    }

    gfx::Instance instance = {};
    result = gfx::CreateInstance(&instance, {
        .minimum_api_version = (u32)VK_API_VERSION_1_1,
        .enable_validation_layer = true,
        .enable_synchronization_validation = true,
        // .enable_gpu_based_validation = true,
    });
    if (result != gfx::Result::SUCCESS) {
        logging::error("sync", "Failed to create vulkan instance\n");
        exit(100);
    }

    gfx::Device device = {};
    result = gfx::CreateDevice(&device, instance, {
        .minimum_api_version = (u32)VK_API_VERSION_1_1,
        .required_features = gfx::DeviceFeatures::DYNAMIC_RENDERING | gfx::DeviceFeatures::DESCRIPTOR_INDEXING | gfx::DeviceFeatures::SYNCHRONIZATION_2,
    });
    if (result != gfx::Result::SUCCESS) {
        logging::error("sync", "Failed to create vulkan device\n");
        exit(100);
    }

    gfx::Window window = {};
    result = gfx::CreateWindowWithSwapchain(&window, instance, device, {
        .title = "XPG",
        .width = 1600,
        .height = 900,
        .preferred_frames_in_flight = 2,
    });
    if (result != gfx::Result::SUCCESS) {
        logging::error("sync", "Failed to create vulkan window\n");
        exit(100);
    }

    struct App {
        // Swapchain frames, index wraps around at the number of frames in flight.
        u32 frame_index;
        // Total frame index.
        u64 current_frame;

        bool force_swapchain_update;
        bool wait_for_events;
        bool closed;
    };

    VkResult vkr;

    App app = {};
    app.wait_for_events = false;

    auto Draw = [&app, &device, &window] () {
        if (app.closed) return;

        gfx::SwapchainStatus swapchain_status = gfx::UpdateSwapchain(&window, device);
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
        }


        // Acquire current frame
        gfx::Frame& frame = gfx::WaitForFrame(&window, device);
        gfx::Result ok = gfx::AcquireImage(&frame, &window, device);
        if (ok != gfx::Result::SUCCESS) {
            return;
        }

        gfx::BeginCommands(frame.command_pool, frame.command_buffer, device);

        // Invalidate caches on the GPU (make visible)
        // NOTE: I dont' think any memory barrier is needed here,
        // because submitting to the queue already counts as one.
        gfx::CmdBarriers(frame.command_buffer, {
            .image = {
                // {
                //     .image = frame.current_image,
                //     .src_stage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                //     .dst_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                //     .src_access = 0,
                //     .dst_access = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                //     .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
                //     .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                // },
                {
                    .src_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    // .src_stage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                    .src_access = 0,
                    .dst_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    .dst_access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    .image = frame.current_image,
                },
            },
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
            .width = window.fb_width,
            .height = window.fb_height,
        });

        gfx::CmdEndRendering(frame.command_buffer);

        gfx::CmdImageBarrier(frame.command_buffer, {
            .src_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .src_access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .dst_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            // .dst_stage = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
            .dst_access = 0,
            .old_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .new_layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .image = frame.current_image,
        });

        gfx::EndCommands(frame.command_buffer);

        VkResult vkr;
        vkr = gfx::Submit1(frame, device,
            // VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
        );
        assert(vkr == VK_SUCCESS);
        vkr = gfx::PresentFrame(&window, &frame, device);
        assert(vkr == VK_SUCCESS);

        app.frame_index = (app.frame_index + 1) % window.images.length;
        app.current_frame += 1;
    };

    gfx::SetWindowCallbacks(&window, {
        .draw = Draw,
    });


    while (true) {
        gfx::ProcessEvents(app.wait_for_events);

        if (gfx::ShouldClose(window)) {
            logging::info("sync", "Window closed");
            app.closed = true;
            break;
        }

        // Draw
        Draw();
    };

    // Wait
    gfx::WaitIdle(device);

    // Window
    gfx::DestroyWindowWithSwapchain(&window, instance, device);

    // Device
    gfx::DestroyDevice(&device);

    // Instance
    gfx::DestroyInstance(&instance);
}
