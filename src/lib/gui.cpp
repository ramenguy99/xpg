#include <xpg/gui.h>

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

namespace gui {

void CreateImGuiImpl(ImGuiImpl* impl, const gfx::Window& window, const gfx::Context& vk, const Config&& config) {
    // Create descriptor pool for imgui.
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
    };

    VkDescriptorPoolCreateInfo descriptor_pool_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    descriptor_pool_info.flags = 0;
    descriptor_pool_info.maxSets = 1;
    descriptor_pool_info.pPoolSizes = pool_sizes;
    descriptor_pool_info.poolSizeCount = ArrayCount(pool_sizes);

    VkDescriptorPool descriptor_pool = 0;
    vkCreateDescriptorPool(vk.device, &descriptor_pool_info, 0, &descriptor_pool);

    // Initialize ImGui.
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    if (!config.enable_ini_and_log_files) {
        io.LogFilename = NULL;
        io.IniFilename = NULL;
    }

    for (usize i = 0; i < config.additional_fonts.length; i++) {
        const Font& font = config.additional_fonts[i];

        ImGuiIO& io = ImGui::GetIO();
        ImFontConfig font_cfg;
        font_cfg.FontDataOwnedByAtlas = font.owned_by_atlas;
        io.Fonts->AddFontFromMemoryTTF(font.data.data, (int)font.data.length, font.size, &font_cfg);
    }

    if (!ImGui_ImplGlfw_InitForVulkan(window.window, true)) {
        printf("Failed to initialize ImGui\n");
        exit(1);
    }

    // TODO: MSAA
    ImGui_ImplVulkan_InitInfo vk_init_info = {};
    vk_init_info.Instance = vk.instance;
    vk_init_info.PhysicalDevice = vk.physical_device;
    vk_init_info.Device = vk.device;
    vk_init_info.QueueFamily = vk.queue_family_index;
    vk_init_info.Queue = vk.queue;
    vk_init_info.PipelineCache = 0;
    vk_init_info.DescriptorPool = descriptor_pool;
    vk_init_info.Subpass = 0;
    vk_init_info.MinImageCount = (u32)window.images.length;
    vk_init_info.ImageCount = (u32)window.images.length;
    vk_init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    struct ImGuiCheckResult {
        static void fn(VkResult res) {
            assert(res == VK_SUCCESS);
        }
    };
    vk_init_info.CheckVkResultFn = ImGuiCheckResult::fn;

    VkRenderPass render_pass = VK_NULL_HANDLE;
    if(config.dynamic_rendering) {
        vk_init_info.UseDynamicRendering = true;
        vk_init_info.ColorAttachmentFormat = window.swapchain_format;
    } else {
        VkAttachmentDescription attachments[1];
        attachments[0].format = window.swapchain_format;
        attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        attachments[0].flags = 0;

        VkAttachmentReference color_reference = {};
        color_reference.attachment = 0;
        color_reference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.flags = 0;
        subpass.inputAttachmentCount = 0;
        subpass.pInputAttachments = NULL;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color_reference;
        subpass.pResolveAttachments = NULL;
        subpass.pDepthStencilAttachment = NULL;
        subpass.preserveAttachmentCount = 0;
        subpass.pPreserveAttachments = NULL;

        VkSubpassDependency subpass_dependency = {};
        subpass_dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        subpass_dependency.dstSubpass = 0;
        subpass_dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        subpass_dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        subpass_dependency.srcAccessMask = 0;
        subpass_dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        subpass_dependency.dependencyFlags = 0;

        VkRenderPassCreateInfo rp_info = {};
        rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rp_info.pNext = NULL;
        rp_info.attachmentCount = 1;
        rp_info.pAttachments = attachments;
        rp_info.subpassCount = 1;
        rp_info.pSubpasses = &subpass;
        rp_info.dependencyCount = 1;
        rp_info.pDependencies = &subpass_dependency;

        VkResult vkr = vkCreateRenderPass(vk.device, &rp_info, NULL, &render_pass);
        assert(vkr == VK_SUCCESS);
    }

    if (!ImGui_ImplVulkan_Init(&vk_init_info, render_pass)) {
        printf("Failed to initialize Vulkan imgui backend\n");
        exit(1);
    }

    // Upload font texture.
    VkCommandPool command_pool = window.frames[0].command_pool;
    VkCommandBuffer command_buffer = window.frames[0].command_buffer;

    // Reset command buffer.
    VkResult vkr = vkResetCommandPool(vk.device, command_pool, 0);
    assert(vkr == VK_SUCCESS);

    // Begin recording commands.
    VkCommandBufferBeginInfo begin_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkr = vkBeginCommandBuffer(command_buffer, &begin_info);
    assert(vkr == VK_SUCCESS);

    // Create fonts texture.
    ImGui_ImplVulkan_CreateFontsTexture(command_buffer);

    // End recording commands.
    vkr = vkEndCommandBuffer(command_buffer);
    assert(vkr == VK_SUCCESS);

    // Submit commands.
    VkPipelineStageFlags submit_stage_mask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

    VkSubmitInfo submit_info = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;

    vkr = vkQueueSubmit(vk.queue, 1, &submit_info, VK_NULL_HANDLE);
    assert(vkr == VK_SUCCESS);

    // Wait for idle.
    vkr = vkDeviceWaitIdle(vk.device);
    assert(vkr == VK_SUCCESS);


    impl->descriptor_pool = descriptor_pool;
    impl->render_pass = render_pass;
}

void DestroyImGuiImpl(ImGuiImpl* impl, gfx::Context& vk) {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    vkDestroyDescriptorPool(vk.device, impl->descriptor_pool, 0);
    vkDestroyRenderPass(vk.device, impl->render_pass, 0);
}

void BeginFrame() {
    ImGui_ImplGlfw_NewFrame();
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();
}

void Render(VkCommandBuffer cmd)
{
    ImDrawData* draw_data = ImGui::GetDrawData();
    ImGui_ImplVulkan_RenderDrawData(draw_data, cmd);
}

void EndFrame() {
    ImGui::Render();
}

void DrawStats(f32 SecondsElapsed, u32 width, u32 height)
{
    static f32 FrameTimes[32] = {};
    static int FrameTimesIndex = 0;

    FrameTimes[FrameTimesIndex++] = SecondsElapsed;
    FrameTimesIndex %= ArrayCount(FrameTimes);

    f32 AverageSecondsElapsed = 0.0f;
    for (u32 Index = 0; Index < ArrayCount(FrameTimes); Index++)
    {
        AverageSecondsElapsed += FrameTimes[Index];
    }
    AverageSecondsElapsed /= ArrayCount(FrameTimes);

    static bool Show = true;
    bool* p_open = &Show;
    const float DISTANCE = 10.0f;
    static int corner = 0;
    ImVec2 window_pos = ImVec2((corner & 1) ? ImGui::GetIO().DisplaySize.x - DISTANCE : DISTANCE, (corner & 2) ? ImGui::GetIO().DisplaySize.y - DISTANCE : DISTANCE);
    ImVec2 window_pos_pivot = ImVec2((corner & 1) ? 1.0f : 0.0f, (corner & 2) ? 1.0f : 0.0f);
    ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
    ImGui::SetNextWindowBgAlpha(0.5f); // Transparent background
    if (ImGui::Begin("Debug Data", p_open, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav))
    {
        ImGui::Text("FPS: %.3f", 1.0f / AverageSecondsElapsed);
        ImGui::Text("Frame Time: %.3fms", AverageSecondsElapsed * 1000.0f);
        ImGui::Separator();
        ImGui::Text("Window size: (%d, %.d)", width, height);
        ImGui::End();
    }
}

}
