
#include <xpg/gui.h>

using glm::vec2;
using glm::ivec2;
using glm::vec3;
using glm::vec4;
using glm::mat4;

int main(int argc, char** argv) {
    gfx::Result result;
    result = gfx::Init();
    if (result != gfx::Result::SUCCESS) {
        logging::error("raytrace", "Failed to initialize platform\n");
    }

    Array<const char*> instance_extensions = gfx::GetPresentationInstanceExtensions();
    instance_extensions.add("VK_EXT_debug_report");

    Array<const char*> device_extensions;
    device_extensions.add("VK_KHR_swapchain");
    device_extensions.add("VK_KHR_dynamic_rendering");
    device_extensions.add("VK_KHR_deferred_host_operations");
    device_extensions.add("VK_KHR_acceleration_structure");
    device_extensions.add("VK_KHR_ray_query");
    device_extensions.add("VK_KHR_ray_tracing_pipeline");

    gfx::Context vk = {};
    result = gfx::CreateContext(&vk, {
        .minimum_api_version = (u32)VK_API_VERSION_1_3,
        .instance_extensions = instance_extensions,
        .device_extensions = device_extensions,
        .device_features = gfx::DeviceFeatures::DYNAMIC_RENDERING | gfx::DeviceFeatures::DESCRIPTOR_INDEXING | gfx::DeviceFeatures::SYNCHRONIZATION_2 | gfx::DeviceFeatures::RAYTRACING,
        .enable_validation_layer = true,
        //        .enable_gpu_based_validation = true,
    });

    if (result != gfx::Result::SUCCESS) {
        logging::error("raytrace", "Failed to initialize vulkan");
        exit(100);
    }

    gfx::Window window = {};
    result = gfx::CreateWindowWithSwapchain(&window, vk, "raytrace", 1600, 900);
    if (result != gfx::Result::SUCCESS) {
        logging::error("raytrace", "Failed to create vulkan window");
        exit(100);
    }

    VkResult vkr;

    Array<u8> shader_code;
    if (platform::ReadEntireFile("res/raytrace.comp.spirv", &shader_code) != platform::Result::Success) {
        logging::error("raytrace", "Failed to read compute shader");
        exit(100);
    }

    // TODO: describe geometry
    ArrayFixed<vec3, 3> vertices(3);
    vertices[0] = vec3(-0.5, -0.5, 0);
    vertices[1] = vec3(   0,  0.5, 0);
    vertices[2] = vec3( 0.5, -0.5, 0);

    gfx::Buffer vertices_buffer;
    vkr = gfx::CreateBufferFromData(&vertices_buffer, vk, vertices.as_bytes(), {
        .usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        .alloc = gfx::AllocPresets::DeviceMapped,
    });
    if (vkr != VK_SUCCESS) {
        logging::error("raytrace", "Failed to create vertices buffer");
        exit(100);
    }

    ArrayFixed<u32, 3> indices(3);
    indices[0] = 0;
    indices[1] = 2;
    indices[2] = 1;
    gfx::Buffer indices_buffer;
    vkr = gfx::CreateBufferFromData(&indices_buffer, vk, indices.as_bytes(), {
        .usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        .alloc = gfx::AllocPresets::DeviceMapped,
    });
    if (vkr != VK_SUCCESS) {
        logging::error("raytrace", "Failed to create indices buffer");
        exit(100);
    }


    gfx::AccelerationStructure as = {};
    vkr = gfx::CreateAccelerationStructure(&as, vk, {
        .meshes = {
            {
                .vertices_address = gfx::GetBufferAddress(vertices_buffer.buffer, vk.device),
                .vertices_stride = sizeof(vertices[0]),
                .vertices_count = (u32)vertices.length,
                .vertices_format = VK_FORMAT_R32G32B32_SFLOAT,
                .indices_address = gfx::GetBufferAddress(indices_buffer.buffer, vk.device),
                .indices_type = VK_INDEX_TYPE_UINT32,
                .primitive_count = (u32)(indices.length / 3),
                .transform = glm::mat3x4(1.0),
            },
        },
    });
    if (vkr != VK_SUCCESS) {
        logging::error("raytrace", "Failed to create acceleration structure");
        exit(100);
    }

    gfx::Image output_image;
    vkr = gfx::CreateImage(&output_image, vk, {
        .width = window.fb_width,
        .height = window.fb_height,
        .format = VK_FORMAT_R32G32B32A32_SFLOAT,
        .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .alloc = gfx::AllocPresets::DeviceDedicated,
    });
    if (vkr != VK_SUCCESS) {
        logging::error("raytrace", "Failed to create output image");
        exit(100);
    }

    struct Constants {
        u32 width;
        u32 height;
    };

    Array<gfx::Buffer> constant_buffers(window.frames.length);
    Array<gfx::DescriptorSet> descriptor_sets(window.frames.length);
    for(usize i = 0; i < descriptor_sets.length; i++) {
        vkr = gfx::CreateBuffer(&constant_buffers[i], vk, sizeof(Constants), {
            .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            .alloc = gfx::AllocPresets::DeviceMapped,
        });
        if (vkr != VK_SUCCESS) {
            logging::error("raytrace", "Failed to create uniform buffer");
            exit(100);
        }

        vkr = gfx::CreateDescriptorSet(&descriptor_sets[i], vk, {
            .entries = {
                {
                    .count = 1,
                    .type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                },
                {
                    .count = 1,
                    .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                },
                {
                    .count = 1,
                    .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                },
            }
        });

        if (vkr != VK_SUCCESS) {
            logging::error("raytrace", "Failed to create descriptor set");
            exit(100);
        }

        gfx::WriteAccelerationStructureDescriptor(descriptor_sets[i].set, vk, {
            .acceleration_structure = as.tlas,
            .binding = 0,
        });

        gfx::WriteImageDescriptor(descriptor_sets[i].set, vk, {
            .view = output_image.view,
            .layout = VK_IMAGE_LAYOUT_GENERAL,
            .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .binding = 1,
        });

        gfx::WriteBufferDescriptor(descriptor_sets[i].set, vk, {
            .buffer = constant_buffers[0].buffer,
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .binding = 2,
        });
    }

    gfx::Shader shader;
    vkr = gfx::CreateShader(&shader, vk, shader_code);
    if (vkr != VK_SUCCESS) {
        logging::error("raytrace", "Failed to create compute shader");
        exit(100);
    }

    gfx::ComputePipeline compute_pipeline;
    vkr = gfx::CreateComputePipeline(&compute_pipeline, vk, {
        .shader = shader,
        .descriptor_sets = { descriptor_sets[0].layout },
    });
    if (vkr != VK_SUCCESS) {
        logging::error("raytrace", "Failed to create compute pipeline");
        exit(100);
    }

    struct App {
        // - Window
        bool wait_for_events = true;
        bool closed = false;
        bool first_frame_done = false;

        // - UI
        platform::Timestamp last_frame_timestamp;
        gui::ImGuiImpl gui;

        // - Scene

        // - Rendering
        gfx::AccelerationStructure as;
        gfx::Image output_image;
        Array<gfx::Buffer> constant_buffers;
        Array<gfx::DescriptorSet> descriptor_sets;
        gfx::ComputePipeline compute_pipeline;
        u32 frame_index = 0; // Rendering frame index, wraps around at the number of frames in flight
    };

    // USER: application
    App app = {};
    app.last_frame_timestamp = platform::GetTimestamp();
    app.compute_pipeline = compute_pipeline;
    app.descriptor_sets = std::move(descriptor_sets);
    app.as = std::move(as);
    app.output_image = output_image;
    app.constant_buffers = std::move(constant_buffers);

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
            logging::error("raytrace/draw", "Swapchain update failed\n");
            exit(101);
        }
        else if (swapchain_status == gfx::SwapchainStatus::MINIMIZED) {
            return;
        }

        if (swapchain_status == gfx::SwapchainStatus::RESIZED || !app.first_frame_done) {
            app.first_frame_done = true;
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


            // Record commands
            gfx::BeginCommands(frame.command_pool, frame.command_buffer, vk);

            gfx::CmdBarriers(frame.command_buffer, {
                .image = {
                    {
                        .image = app.output_image.image,
                        .src_stage = VK_PIPELINE_STAGE_2_NONE,
                        .dst_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        .src_access = 0,
                        .dst_access = VK_ACCESS_2_SHADER_WRITE_BIT,
                        .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
                        .new_layout = VK_IMAGE_LAYOUT_GENERAL,
                    },
                    {
                        .image = frame.current_image,
                        .src_stage = VK_PIPELINE_STAGE_2_NONE,
                        .dst_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .src_access = 0,
                        .dst_access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
                        .new_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    },
                }
            });

            Constants constants;
            constants.width = window.fb_width;
            constants.height = window.fb_height;
            app.constant_buffers[app.frame_index].map.as_bytes().copy_exact(BytesOf(&constants));

            vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, app.compute_pipeline.pipeline);
            vkCmdBindDescriptorSets(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, app.compute_pipeline.layout, 0, 1, &app.descriptor_sets[app.frame_index].set, 0, 0);
            vkCmdDispatch(frame.command_buffer, DivCeil(window.fb_width, 32), DivCeil(window.fb_height, 32), 1);

            gfx::CmdBarriers(frame.command_buffer, {
                .image = {
                    {
                        .image = app.output_image.image,
                        .src_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        .dst_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .src_access = VK_ACCESS_2_SHADER_WRITE_BIT,
                        .dst_access = VK_ACCESS_2_TRANSFER_READ_BIT,
                        .old_layout = VK_IMAGE_LAYOUT_GENERAL,
                        .new_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    },
                }
            });

            VkImageBlit region = {};
            region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.srcSubresource.layerCount = 1;
            region.srcOffsets[1].x = window.fb_width;
            region.srcOffsets[1].y = window.fb_height;
            region.srcOffsets[1].z = 1;
            region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.dstSubresource.layerCount = 1;
            region.dstOffsets[1].x = window.fb_width;
            region.dstOffsets[1].y = window.fb_height;
            region.dstOffsets[1].z = 1;
            vkCmdBlitImage(frame.command_buffer, app.output_image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, frame.current_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_NEAREST);

            gfx::CmdBarriers(frame.command_buffer, {
                .image = {
                    {
                        .image = frame.current_image,
                        .src_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .dst_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        .src_access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        .dst_access = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                        .old_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    },
                }
            });

            gfx::CmdBeginRendering(frame.command_buffer, {
                .color = {
                    {
                        .view = frame.current_image_view,
                        .load_op = VK_ATTACHMENT_LOAD_OP_LOAD,
                        .store_op = VK_ATTACHMENT_STORE_OP_STORE,
                    },
                },
                .width = window.fb_width,
                .height = window.fb_height,
            });

            gui::Render(frame.command_buffer);
            vkCmdEndRenderingKHR(frame.command_buffer);

            gfx::CmdImageBarrier(frame.command_buffer, {
                .image = frame.current_image,
                .src_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .dst_stage = VK_PIPELINE_STAGE_2_NONE,
                .src_access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
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

        app.frame_index = (app.frame_index + 1) % (u32)window.frames.length;
    };

    gfx::SetWindowCallbacks(&window, {
        .mouse_move_event = MouseMoveEvent,
        .mouse_button_event = MouseButtonEvent,
        .mouse_scroll_event = MouseScrollEvent,
        .draw = Draw,
    });

    gui::CreateImGuiImpl(&app.gui, window, vk, {});

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
    gfx::DestroyAccelerationStructure(&app.as, vk);
    gfx::DestroyBuffer(&vertices_buffer, vk);
    gfx::DestroyBuffer(&indices_buffer, vk);
    gfx::DestroyShader(&shader, vk);
    gfx::DestroyComputePipeline(&app.compute_pipeline, vk);
    gfx::DestroyImage(&app.output_image, vk);
    for(usize i = 0; i < app.descriptor_sets.length; i++) {
        gfx::DestroyBuffer(&app.constant_buffers[i], vk);
        gfx::DestroyDescriptorSet(&app.descriptor_sets[i], vk);
    }

    // Gui
    gui::DestroyImGuiImpl(&app.gui, vk);

    // Window
    gfx::DestroyWindowWithSwapchain(&window, vk);

    // Context
    gfx::DestroyContext(&vk);
}
