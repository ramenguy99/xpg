namespace gui {

struct ImGuiImpl {
    VkDescriptorPool descriptor_pool = 0;
};

void Create(ImGuiImpl* impl, const gfx::Window& window, const gfx::Context& vk) {
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
    vk_init_info.UseDynamicRendering = true;
    vk_init_info.ColorAttachmentFormat = window.swapchain_format;
    struct ImGuiCheckResult {
        static void fn(VkResult res) {
            assert(res == VK_SUCCESS);
        }
    };
    vk_init_info.CheckVkResultFn = ImGuiCheckResult::fn;

    if (!ImGui_ImplVulkan_Init(&vk_init_info, VK_NULL_HANDLE)) {
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
}

void Destroy(ImGuiImpl* impl, gfx::Context& vk) {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    vkDestroyDescriptorPool(vk.device, impl->descriptor_pool, 0);
}

void BeginFrame() {
    ImGui_ImplGlfw_NewFrame();
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();
}

void EndFrame() {
    ImGui::Render();
}

}
