#include <stdio.h>

#include <slang-gfx.h>
#include <slang-com-ptr.h>
#include <slang.h>

#include "reflection_printing.h"

#include <xpg/defines.h>
#include <xpg/log.h>
#include <xpg/platform.h>

using namespace slang;
using namespace xpg;

// const char* shortestShader =
// "RWStructuredBuffer<float> result;"
// "[shader(\"compute\")]"
// "[numthreads(1,1,1)]"
// "void computeMain(uint3 threadId : SV_DispatchThreadID)"
// "{"
// "    result[threadId.x] = threadId.x;"
// "}";

int main(int argc, char** argv) {
    if (argc < 2)
    {
        logging::error("shader_reflection", "Expected 1 argument got 0");
        exit(1);
    }

    Array<u8> source;
    platform::Result res = platform::ReadEntireFile(argv[1], &source);
    if (res != platform::Result::Success)
    {
        logging::error("shader_reflection", "Failed to open file: %s (Error code: %d)", argv[1], (int)res);
        exit(1);
    }
    source.add(0);
    Slang::ComPtr<IGlobalSession> globalSession;

    createGlobalSession(globalSession.writeRef());

    slang::SessionDesc sessionDesc = {};
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_SPIRV;
    targetDesc.profile = globalSession->findProfile("spirv_1_6");

    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;

    slang::PreprocessorMacroDesc preprocessorMacroDesc[] = {
        { "__XPG", "1" },
    };

    sessionDesc.preprocessorMacros = preprocessorMacroDesc;
    sessionDesc.preprocessorMacroCount = ArrayCount(preprocessorMacroDesc);

    slang::CompilerOptionEntry options[] =
    {
        {
            slang::CompilerOptionName::EmitSpirvDirectly,
            {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr}
        }
    };
    sessionDesc.compilerOptionEntries = options;
    sessionDesc.compilerOptionEntryCount = ArrayCount(options);

    Slang::ComPtr<slang::ISession> session;
    globalSession->createSession(sessionDesc, session.writeRef());

    Slang::ComPtr<slang::IModule> slangModule;
    {
        Slang::ComPtr<slang::IBlob> diagnosticsBlob;
        const char* moduleName = "shortest";
        const char* modulePath = "shortest.slang";
        slangModule = session->loadModuleFromSourceString(moduleName, modulePath, (char*)source.data, diagnosticsBlob.writeRef());
        diagnoseIfNeeded(diagnosticsBlob);
        if (!slangModule) {
            exit(1);
        }
    }

    //  // 4. Query Entry Points
    // Slang::ComPtr<slang::IEntryPoint> entryPoint;
    // {
    //     Slang::ComPtr<slang::IBlob> diagnosticsBlob;
    //     slangModule->findEntryPointByName("computeMain", entryPoint.writeRef());
    //     if (!entryPoint)
    //     {
    //         logging::error("shader_reflection", "Error getting entry point %llu", slangModule->getDefinedEntryPointCount());
    //         exit(1);
    //     }
    // }

    // // 5. Compose Modules + Entry Points
    // slang::IComponentType* componentTypes[] = {
    //     slangModule,
    //     entryPoint
    // };

    // Slang::ComPtr<slang::IComponentType> composedProgram;
    // {
    //     Slang::ComPtr<slang::IBlob> diagnosticsBlob;
    //     SlangResult result = session->createCompositeComponentType(
    //         componentTypes,
    //         ArrayCount(componentTypes),
    //         composedProgram.writeRef(),
    //         diagnosticsBlob.writeRef());
    //     diagnoseIfNeeded(diagnosticsBlob);
    //     if(SLANG_FAILED(result)) {
    //         logging::error("shader_reflection", "Error getting entry point");
    //         exit(1);
    //     }
    // }

    // // 6. Link
    // Slang::ComPtr<slang::IComponentType> linkedProgram;
    // {
    //     Slang::ComPtr<slang::IBlob> diagnosticsBlob;
    //     SlangResult result = composedProgram->link(
    //         linkedProgram.writeRef(),
    //         diagnosticsBlob.writeRef());
    //     diagnoseIfNeeded(diagnosticsBlob);
    //     if(SLANG_FAILED(result)) {
    //         logging::error("shader_reflection", "Error linking");
    //         exit(1);
    //     }
    // }

    // // 7. Get Target Kernel Code
    // Slang::ComPtr<slang::IBlob> spirvCode;
    // {
    //     Slang::ComPtr<slang::IBlob> diagnosticsBlob;
    //     SlangResult result = linkedProgram->getEntryPointCode(
    //         0,
    //         0,
    //         spirvCode.writeRef(),
    //         diagnosticsBlob.writeRef());
    //     diagnoseIfNeeded(diagnosticsBlob);
    //     if(SLANG_FAILED(result)) {
    //         logging::error("shader_reflection", "Error getting spirv code");
    //         exit(1);
    //     }
    // }
    // logging::info("shader_reflection", "Compiled %llu bytes of SPIR-V", spirvCode->getBufferSize());

    // 8. Get layout
    // slang::ProgramLayout* programLayout = linkedProgram->getLayout();

    ReflectingPrinting printer;
    printer.compileAndReflectModule(session, slangModule);
    printf("\n");
    // system("pause");
}
/*
    #include <xpg/gui.h>

    gfx::Result result;
    result = gfx::Init();
    if (result != gfx::Result::SUCCESS) {
        logging::error("shader_reflection", "Failed to initialize platform\n");
    }

    gfx::Context vk = {};
    result = gfx::CreateContext(&vk, {
        .minimum_api_version = (u32)VK_API_VERSION_1_1,
        .required_features = gfx::DeviceFeatures::DYNAMIC_RENDERING | gfx::DeviceFeatures::DESCRIPTOR_INDEXING | gfx::DeviceFeatures::SYNCHRONIZATION_2,
        .enable_validation_layer = true,
        //        .enable_gpu_based_validation = true,
    });
    if (result != gfx::Result::SUCCESS) {
        logging::error("shader_reflection", "Failed to initialize vulkan\n");
        exit(100);
    }

    gfx::Window window = {};
    result = gfx::CreateWindowWithSwapchain(&window, vk, "XPG", 1600, 900);
    if (result != gfx::Result::SUCCESS) {
        logging::error("shader_reflection", "Failed to create vulkan window\n");
        exit(100);
    }

    struct App {
        // - Window
        bool wait_for_events = true;
        bool closed = false;
        bool first_frame_done = false;

        // - UI
        platform::Timestamp last_frame_timestamp;

        // - Render
        u64 frame_index = 0;
    };

    // USER: application
    App app = {};
    app.last_frame_timestamp = platform::GetTimestamp();

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
        }

        // Acquire current frame
        gfx::Frame& frame = gfx::WaitForFrame(&window, vk);
        gfx::Result ok = gfx::AcquireImage(&frame, &window, vk);
        if (ok != gfx::Result::SUCCESS) {
            return;
        }

        {
            gfx::BeginCommands(frame.command_pool, frame.command_buffer, vk);

            {
                gui::BeginFrame();

                // USER: gui
                gui::DrawStats(dt, window.fb_width, window.fb_height);

                gui::EndFrame();
            }


            // USER: draw commands
            gfx::CmdImageBarrier(frame.command_buffer, frame.current_image, {
                .src_stage = VK_PIPELINE_STAGE_2_NONE,
                .dst_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .src_access = 0,
                .dst_access = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
                .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                });

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

            // Draw GUI
            gui::Render(frame.command_buffer);

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

        app.frame_index = (app.frame_index + 1) % (u32)window.frames.length;
    };

    gfx::SetWindowCallbacks(&window, {
        .draw = Draw,
    });

    gui::ImGuiImpl imgui_impl;
    gui::CreateImGuiImpl(&imgui_impl, window, vk, {});

    while (true) {
        gfx::ProcessEvents(app.wait_for_events);

        if (gfx::ShouldClose(window)) {
            logging::info("shader_reflection", "Window closed");
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
    gui::DestroyImGuiImpl(&imgui_impl, vk);

    // Window
    gfx::DestroyWindowWithSwapchain(&window, vk);

    // Context
    gfx::DestroyContext(&vk);
}
*/
