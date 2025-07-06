#pragma once

#include "gfx.h"

#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui.h>

namespace xpg {
namespace gui {

//- Impl
struct ImGuiImpl {
    VkRenderPass render_pass = VK_NULL_HANDLE;
};

struct Font {
    ArrayView<u8> data;
    float size = 12.0f;
    bool owned_by_atlas = false;
};

struct Config {
    bool dynamic_rendering = true;
    bool enable_ini_and_log_files = true;
    Span<Font> additional_fonts;
};

void CreateImGuiImpl(ImGuiImpl* impl, const gfx::Window& window, const gfx::Context& vk, const Config&& config);
void DestroyImGuiImpl(ImGuiImpl* impl, gfx::Context& vk);

//- Frames
void BeginFrame();
void Render(VkCommandBuffer cmd);
void EndFrame();

//- GUIs
void DrawStats(f32 SecondsElapsed, u32 width, u32 height);

//- Style
void SetLightTheme();
void SetDarkTheme();

} // namespace gui
} // namespace xpg
