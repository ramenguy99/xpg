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

namespace xpg {
namespace gui {

void CreateImGuiImpl(ImGuiImpl* impl, const gfx::Window& window, const gfx::Context& vk, const Config&& config) {
    // Initialize ImGui.
    ImGui::CreateContext();

    if (!ImGui_ImplGlfw_InitForVulkan(window.window, true)) {
        printf("Failed to initialize ImGui\n");
        exit(1);
    }

    // TODO: MSAA
    ImGui_ImplVulkan_InitInfo vk_init_info = {};
    vk_init_info.ApiVersion = vk.instance_version;
    vk_init_info.Instance = vk.instance;
    vk_init_info.PhysicalDevice = vk.physical_device;
    vk_init_info.Device = vk.device;
    vk_init_info.QueueFamily = vk.queue_family_index;
    vk_init_info.Queue = vk.queue;
    vk_init_info.DescriptorPool = VK_NULL_HANDLE;
    vk_init_info.RenderPass = VK_NULL_HANDLE;
    vk_init_info.MinImageCount = (u32)window.images.length;
    vk_init_info.ImageCount = (u32)window.images.length;
    vk_init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    vk_init_info.PipelineCache = VK_NULL_HANDLE;
    vk_init_info.Subpass = 0;
    vk_init_info.DescriptorPoolSize = IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE + 8;
    struct ImGuiCheckResult {
        static void fn(VkResult res) {
            assert(res == VK_SUCCESS);
        }
    };
    vk_init_info.CheckVkResultFn = ImGuiCheckResult::fn;

    VkRenderPass render_pass = VK_NULL_HANDLE;
    if(config.dynamic_rendering) {
        vk_init_info.UseDynamicRendering = true;
        vk_init_info.PipelineRenderingCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR };
        vk_init_info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
        vk_init_info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &window.swapchain_format;
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

        vk_init_info.RenderPass = render_pass;
    }

    if (!ImGui_ImplVulkan_Init(&vk_init_info)) {
        printf("Failed to initialize Vulkan imgui backend\n");
        exit(1);
    }

    // Add fonts
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
    ImGui_ImplVulkan_CreateFontsTexture();

    impl->render_pass = render_pass;
}

void DestroyImGuiImpl(ImGuiImpl* impl, gfx::Context& vk) {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

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
    if(draw_data) {
        ImGui_ImplVulkan_RenderDrawData(draw_data, cmd);
    }
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

namespace Spectrum {
    namespace { // Unnamed namespace, since we only use this here.
        unsigned int Color(unsigned int c) {
            // add alpha and swap red and blue channel
            const short a = 0xFF;
            const short r = (c >> 16) & 0xFF;
            const short g = (c >> 8) & 0xFF;
            const short b = (c >> 0) & 0xFF;
            return(a << 24)
                | (r << 0)
                | (g << 8)
                | (b << 16);
        }
    }

    namespace Light {
        const unsigned int NONE = 0x00000000; // transparent
        const unsigned int GRAY50 = Color(0xFFFFFF);
        const unsigned int GRAY75 = Color(0xFAFAFA);
        const unsigned int GRAY100 = Color(0xF5F5F5);
        const unsigned int GRAY200 = Color(0xEAEAEA);
        const unsigned int GRAY300 = Color(0xE1E1E1);
        const unsigned int GRAY400 = Color(0xCACACA);
        const unsigned int GRAY500 = Color(0xB3B3B3);
        const unsigned int GRAY600 = Color(0x8E8E8E);
        const unsigned int GRAY700 = Color(0x707070);
        const unsigned int GRAY800 = Color(0x4B4B4B);
        const unsigned int GRAY900 = Color(0x2C2C2C);
        const unsigned int BLUE400 = Color(0x2680EB);
        const unsigned int BLUE500 = Color(0x1473E6);
        const unsigned int BLUE600 = Color(0x0D66D0);
        const unsigned int BLUE700 = Color(0x095ABA);
        const unsigned int RED400 = Color(0xE34850);
        const unsigned int RED500 = Color(0xD7373F);
        const unsigned int RED600 = Color(0xC9252D);
        const unsigned int RED700 = Color(0xBB121A);
        const unsigned int ORANGE400 = Color(0xE68619);
        const unsigned int ORANGE500 = Color(0xDA7B11);
        const unsigned int ORANGE600 = Color(0xCB6F10);
        const unsigned int ORANGE700 = Color(0xBD640D);
        const unsigned int GREEN400 = Color(0x2D9D78);
        const unsigned int GREEN500 = Color(0x268E6C);
        const unsigned int GREEN600 = Color(0x12805C);
        const unsigned int GREEN700 = Color(0x107154);
        const unsigned int INDIGO400 = Color(0x6767EC);
        const unsigned int INDIGO500 = Color(0x5C5CE0);
        const unsigned int INDIGO600 = Color(0x5151D3);
        const unsigned int INDIGO700 = Color(0x4646C6);
        const unsigned int CELERY400 = Color(0x44B556);
        const unsigned int CELERY500 = Color(0x3DA74E);
        const unsigned int CELERY600 = Color(0x379947);
        const unsigned int CELERY700 = Color(0x318B40);
        const unsigned int MAGENTA400 = Color(0xD83790);
        const unsigned int MAGENTA500 = Color(0xCE2783);
        const unsigned int MAGENTA600 = Color(0xBC1C74);
        const unsigned int MAGENTA700 = Color(0xAE0E66);
        const unsigned int YELLOW400 = Color(0xDFBF00);
        const unsigned int YELLOW500 = Color(0xD2B200);
        const unsigned int YELLOW600 = Color(0xC4A600);
        const unsigned int YELLOW700 = Color(0xB79900);
        const unsigned int FUCHSIA400 = Color(0xC038CC);
        const unsigned int FUCHSIA500 = Color(0xB130BD);
        const unsigned int FUCHSIA600 = Color(0xA228AD);
        const unsigned int FUCHSIA700 = Color(0x93219E);
        const unsigned int SEAFOAM400 = Color(0x1B959A);
        const unsigned int SEAFOAM500 = Color(0x16878C);
        const unsigned int SEAFOAM600 = Color(0x0F797D);
        const unsigned int SEAFOAM700 = Color(0x096C6F);
        const unsigned int CHARTREUSE400 = Color(0x85D044);
        const unsigned int CHARTREUSE500 = Color(0x7CC33F);
        const unsigned int CHARTREUSE600 = Color(0x73B53A);
        const unsigned int CHARTREUSE700 = Color(0x6AA834);
        const unsigned int PURPLE400 = Color(0x9256D9);
        const unsigned int PURPLE500 = Color(0x864CCC);
        const unsigned int PURPLE600 = Color(0x7A42BF);
        const unsigned int PURPLE700 = Color(0x6F38B1);
    }

    namespace Dark {
        const unsigned int NONE = 0x00000000; // transparent
        const unsigned int GRAY50 = Color(0x252525);
        const unsigned int GRAY75 = Color(0x2F2F2F);
        const unsigned int GRAY100 = Color(0x323232);
        const unsigned int GRAY200 = Color(0x393939);
        const unsigned int GRAY300 = Color(0x3E3E3E);
        const unsigned int GRAY400 = Color(0x4D4D4D);
        const unsigned int GRAY500 = Color(0x5C5C5C);
        const unsigned int GRAY600 = Color(0x7B7B7B);
        const unsigned int GRAY700 = Color(0x999999);
        const unsigned int GRAY800 = Color(0xCDCDCD);
        const unsigned int GRAY900 = Color(0xFFFFFF);
        const unsigned int BLUE400 = Color(0x2680EB);
        const unsigned int BLUE500 = Color(0x378EF0);
        const unsigned int BLUE600 = Color(0x4B9CF5);
        const unsigned int BLUE700 = Color(0x5AA9FA);
        const unsigned int RED400 = Color(0xE34850);
        const unsigned int RED500 = Color(0xEC5B62);
        const unsigned int RED600 = Color(0xF76D74);
        const unsigned int RED700 = Color(0xFF7B82);
        const unsigned int ORANGE400 = Color(0xE68619);
        const unsigned int ORANGE500 = Color(0xF29423);
        const unsigned int ORANGE600 = Color(0xF9A43F);
        const unsigned int ORANGE700 = Color(0xFFB55B);
        const unsigned int GREEN400 = Color(0x2D9D78);
        const unsigned int GREEN500 = Color(0x33AB84);
        const unsigned int GREEN600 = Color(0x39B990);
        const unsigned int GREEN700 = Color(0x3FC89C);
        const unsigned int INDIGO400 = Color(0x6767EC);
        const unsigned int INDIGO500 = Color(0x7575F1);
        const unsigned int INDIGO600 = Color(0x8282F6);
        const unsigned int INDIGO700 = Color(0x9090FA);
        const unsigned int CELERY400 = Color(0x44B556);
        const unsigned int CELERY500 = Color(0x4BC35F);
        const unsigned int CELERY600 = Color(0x51D267);
        const unsigned int CELERY700 = Color(0x58E06F);
        const unsigned int MAGENTA400 = Color(0xD83790);
        const unsigned int MAGENTA500 = Color(0xE2499D);
        const unsigned int MAGENTA600 = Color(0xEC5AAA);
        const unsigned int MAGENTA700 = Color(0xF56BB7);
        const unsigned int YELLOW400 = Color(0xDFBF00);
        const unsigned int YELLOW500 = Color(0xEDCC00);
        const unsigned int YELLOW600 = Color(0xFAD900);
        const unsigned int YELLOW700 = Color(0xFFE22E);
        const unsigned int FUCHSIA400 = Color(0xC038CC);
        const unsigned int FUCHSIA500 = Color(0xCF3EDC);
        const unsigned int FUCHSIA600 = Color(0xD951E5);
        const unsigned int FUCHSIA700 = Color(0xE366EF);
        const unsigned int SEAFOAM400 = Color(0x1B959A);
        const unsigned int SEAFOAM500 = Color(0x20A3A8);
        const unsigned int SEAFOAM600 = Color(0x23B2B8);
        const unsigned int SEAFOAM700 = Color(0x26C0C7);
        const unsigned int CHARTREUSE400 = Color(0x85D044);
        const unsigned int CHARTREUSE500 = Color(0x8EDE49);
        const unsigned int CHARTREUSE600 = Color(0x9BEC54);
        const unsigned int CHARTREUSE700 = Color(0xA3F858);
        const unsigned int PURPLE400 = Color(0x9256D9);
        const unsigned int PURPLE500 = Color(0x9D64E1);
        const unsigned int PURPLE600 = Color(0xA873E9);
        const unsigned int PURPLE700 = Color(0xB483F0);
    }
}

void SetDarkTheme()
{
    ImGuiStyle* style = &ImGui::GetStyle();
    ImVec4* colors = style->Colors;

    using namespace ImGui;
    using namespace Spectrum::Dark;
    colors[ImGuiCol_Text] = ColorConvertU32ToFloat4(GRAY800); // text on hovered controls is gray900
    colors[ImGuiCol_TextDisabled] = ColorConvertU32ToFloat4(GRAY500);
    colors[ImGuiCol_WindowBg] = ColorConvertU32ToFloat4(GRAY100);
    colors[ImGuiCol_ChildBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg] = ColorConvertU32ToFloat4(GRAY50); // not sure about this. Note: applies to tooltips too.
    colors[ImGuiCol_Border] = ColorConvertU32ToFloat4(GRAY300);
    colors[ImGuiCol_BorderShadow] = ColorConvertU32ToFloat4(NONE); // We don't want shadows. Ever.
    colors[ImGuiCol_FrameBg] = ColorConvertU32ToFloat4(GRAY75); // this isnt right, spectrum does not do this, but it's a good fallback
    colors[ImGuiCol_FrameBgHovered] = ColorConvertU32ToFloat4(GRAY50);
    colors[ImGuiCol_FrameBgActive] = ColorConvertU32ToFloat4(GRAY200);
    colors[ImGuiCol_TitleBg] = ColorConvertU32ToFloat4(GRAY300); // those titlebar values are totally made up, spectrum does not have this.
    colors[ImGuiCol_TitleBgActive] = ColorConvertU32ToFloat4(GRAY200);
    colors[ImGuiCol_TitleBgCollapsed] = ColorConvertU32ToFloat4(GRAY400);
    colors[ImGuiCol_MenuBarBg] = ColorConvertU32ToFloat4(GRAY100);
    colors[ImGuiCol_ScrollbarBg] = ColorConvertU32ToFloat4(GRAY100); // same as regular background
    colors[ImGuiCol_ScrollbarGrab] = ColorConvertU32ToFloat4(GRAY400);
    colors[ImGuiCol_ScrollbarGrabHovered] = ColorConvertU32ToFloat4(GRAY600);
    colors[ImGuiCol_ScrollbarGrabActive] = ColorConvertU32ToFloat4(GRAY700);
    colors[ImGuiCol_CheckMark] = ColorConvertU32ToFloat4(BLUE500);
    colors[ImGuiCol_SliderGrab] = ColorConvertU32ToFloat4(GRAY700);
    colors[ImGuiCol_SliderGrabActive] = ColorConvertU32ToFloat4(GRAY800);
    colors[ImGuiCol_Button] = ColorConvertU32ToFloat4(GRAY75); // match default button to Spectrum's 'Action Button'.
    colors[ImGuiCol_ButtonHovered] = ColorConvertU32ToFloat4(GRAY50);
    colors[ImGuiCol_ButtonActive] = ColorConvertU32ToFloat4(GRAY200);
    colors[ImGuiCol_Header] = ColorConvertU32ToFloat4(BLUE400);
    colors[ImGuiCol_HeaderHovered] = ColorConvertU32ToFloat4(BLUE500);
    colors[ImGuiCol_HeaderActive] = ColorConvertU32ToFloat4(BLUE600);
    colors[ImGuiCol_Separator] = ColorConvertU32ToFloat4(GRAY400);
    colors[ImGuiCol_SeparatorHovered] = ColorConvertU32ToFloat4(GRAY600);
    colors[ImGuiCol_SeparatorActive] = ColorConvertU32ToFloat4(GRAY700);
    colors[ImGuiCol_ResizeGrip] = ColorConvertU32ToFloat4(GRAY400);
    colors[ImGuiCol_ResizeGripHovered] = ColorConvertU32ToFloat4(GRAY600);
    colors[ImGuiCol_ResizeGripActive] = ColorConvertU32ToFloat4(GRAY700);
    colors[ImGuiCol_PlotLines] = ColorConvertU32ToFloat4(BLUE400);
    colors[ImGuiCol_PlotLinesHovered] = ColorConvertU32ToFloat4(BLUE600);
    colors[ImGuiCol_PlotHistogram] = ColorConvertU32ToFloat4(BLUE400);
    colors[ImGuiCol_PlotHistogramHovered] = ColorConvertU32ToFloat4(BLUE600);
    colors[ImGuiCol_TextSelectedBg] = ColorConvertU32ToFloat4((BLUE400 & 0x00FFFFFF) | 0x33000000);
    colors[ImGuiCol_DragDropTarget] = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
    colors[ImGuiCol_NavHighlight] = ColorConvertU32ToFloat4((GRAY900 & 0x00FFFFFF) | 0x0A000000);
    colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.20f, 0.20f, 0.20f, 0.35f);

    // TODO: Improve support for tabs (also add to light theme)
    colors[ImGuiCol_Tab] = ColorConvertU32ToFloat4(GRAY400);                       // Tab background, when tab-bar is focused & tab is unselected
    colors[ImGuiCol_TabHovered] = ColorConvertU32ToFloat4(GRAY500);                // Tab background, when hovered
    colors[ImGuiCol_TabSelected] = ColorConvertU32ToFloat4(GRAY600);               // Tab background, when tab-bar is focused & tab is selected
    colors[ImGuiCol_TabSelectedOverline] = ColorConvertU32ToFloat4(GRAY600);       // Tab horizontal overline, when tab-bar is focused & tab is selected
    colors[ImGuiCol_TabDimmed] = ColorConvertU32ToFloat4(GRAY200);                 // Tab background, when tab-bar is unfocused & tab is unselected
    colors[ImGuiCol_TabDimmedSelected] = ColorConvertU32ToFloat4(GRAY200);         // Tab background, when tab-bar is unfocused & tab is selected
    colors[ImGuiCol_TabDimmedSelectedOverline] = ColorConvertU32ToFloat4(GRAY200); //..horizontal overline, when tab-bar is unfocused & tab is selected
}

void SetLightTheme()
{
    ImGuiStyle* style = &ImGui::GetStyle();
    ImVec4* colors = style->Colors;

    using namespace ImGui;
    using namespace Spectrum::Light;
    colors[ImGuiCol_Text] = ColorConvertU32ToFloat4(GRAY800); // text on hovered controls is gray900
    colors[ImGuiCol_TextDisabled] = ColorConvertU32ToFloat4(GRAY500);
    colors[ImGuiCol_WindowBg] = ColorConvertU32ToFloat4(GRAY100);
    colors[ImGuiCol_ChildBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg] = ColorConvertU32ToFloat4(GRAY50); // not sure about this. Note: applies to tooltips too.
    colors[ImGuiCol_Border] = ColorConvertU32ToFloat4(GRAY300);
    colors[ImGuiCol_BorderShadow] = ColorConvertU32ToFloat4(NONE); // We don't want shadows. Ever.
    colors[ImGuiCol_FrameBg] = ColorConvertU32ToFloat4(GRAY75); // this isnt right, spectrum does not do this, but it's a good fallback
    colors[ImGuiCol_FrameBgHovered] = ColorConvertU32ToFloat4(GRAY50);
    colors[ImGuiCol_FrameBgActive] = ColorConvertU32ToFloat4(GRAY200);
    colors[ImGuiCol_TitleBg] = ColorConvertU32ToFloat4(GRAY300); // those titlebar values are totally made up, spectrum does not have this.
    colors[ImGuiCol_TitleBgActive] = ColorConvertU32ToFloat4(GRAY200);
    colors[ImGuiCol_TitleBgCollapsed] = ColorConvertU32ToFloat4(GRAY400);
    colors[ImGuiCol_MenuBarBg] = ColorConvertU32ToFloat4(GRAY100);
    colors[ImGuiCol_ScrollbarBg] = ColorConvertU32ToFloat4(GRAY100); // same as regular background
    colors[ImGuiCol_ScrollbarGrab] = ColorConvertU32ToFloat4(GRAY400);
    colors[ImGuiCol_ScrollbarGrabHovered] = ColorConvertU32ToFloat4(GRAY600);
    colors[ImGuiCol_ScrollbarGrabActive] = ColorConvertU32ToFloat4(GRAY700);
    colors[ImGuiCol_CheckMark] = ColorConvertU32ToFloat4(BLUE500);
    colors[ImGuiCol_SliderGrab] = ColorConvertU32ToFloat4(GRAY700);
    colors[ImGuiCol_SliderGrabActive] = ColorConvertU32ToFloat4(GRAY800);
    colors[ImGuiCol_Button] = ColorConvertU32ToFloat4(GRAY75); // match default button to Spectrum's 'Action Button'.
    colors[ImGuiCol_ButtonHovered] = ColorConvertU32ToFloat4(GRAY50);
    colors[ImGuiCol_ButtonActive] = ColorConvertU32ToFloat4(GRAY200);
    colors[ImGuiCol_Header] = ColorConvertU32ToFloat4(BLUE400);
    colors[ImGuiCol_HeaderHovered] = ColorConvertU32ToFloat4(BLUE500);
    colors[ImGuiCol_HeaderActive] = ColorConvertU32ToFloat4(BLUE600);
    colors[ImGuiCol_Separator] = ColorConvertU32ToFloat4(GRAY400);
    colors[ImGuiCol_SeparatorHovered] = ColorConvertU32ToFloat4(GRAY600);
    colors[ImGuiCol_SeparatorActive] = ColorConvertU32ToFloat4(GRAY700);
    colors[ImGuiCol_ResizeGrip] = ColorConvertU32ToFloat4(GRAY400);
    colors[ImGuiCol_ResizeGripHovered] = ColorConvertU32ToFloat4(GRAY600);
    colors[ImGuiCol_ResizeGripActive] = ColorConvertU32ToFloat4(GRAY700);
    colors[ImGuiCol_PlotLines] = ColorConvertU32ToFloat4(BLUE400);
    colors[ImGuiCol_PlotLinesHovered] = ColorConvertU32ToFloat4(BLUE600);
    colors[ImGuiCol_PlotHistogram] = ColorConvertU32ToFloat4(BLUE400);
    colors[ImGuiCol_PlotHistogramHovered] = ColorConvertU32ToFloat4(BLUE600);
    colors[ImGuiCol_TextSelectedBg] = ColorConvertU32ToFloat4((BLUE400 & 0x00FFFFFF) | 0x33000000);
    colors[ImGuiCol_DragDropTarget] = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
    colors[ImGuiCol_NavHighlight] = ColorConvertU32ToFloat4((GRAY900 & 0x00FFFFFF) | 0x0A000000);
    colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.20f, 0.20f, 0.20f, 0.35f);
}

} // namespace gui
} // namespace xpg
