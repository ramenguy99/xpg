#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/array.h>

#include <xpg/gfx.h>
#include <xpg/gui.h>

class Context {
public:
    Context() {
        gfx::Result result;
        result = gfx::Init();
        if (result != gfx::Result::SUCCESS) {
            throw std::runtime_error("Failed to initialize platform");
        }

        Array<const char*> instance_extensions = gfx::GetPresentationInstanceExtensions();
        instance_extensions.add("VK_EXT_debug_report");

        Array<const char*> device_extensions;
        device_extensions.add("VK_KHR_swapchain");
        device_extensions.add("VK_KHR_dynamic_rendering");

        result = gfx::CreateContext(&m_vk, {
            .minimum_api_version = (u32)VK_API_VERSION_1_3,
            .instance_extensions = instance_extensions,
            .device_extensions = device_extensions,
            .device_features = gfx::DeviceFeatures::DYNAMIC_RENDERING | gfx::DeviceFeatures::DESCRIPTOR_INDEXING | gfx::DeviceFeatures::SYNCHRONIZATION_2,
            .enable_validation_layer = true,
        });
        if (result != gfx::Result::SUCCESS) {
            throw std::runtime_error("Failed to initialize vulkan");
        }
    }

    ~Context() {
        // TODO: cleanup
    }

private:
    gfx::Context m_vk;
    friend class Window;
};

class Window {
public:
    Window(std::shared_ptr<Context> ctx, const std::string& name, u32 width, u32 height)
        : m_ctx(ctx)
    {
        gfx::Context& vk = ctx->m_vk;

        if (CreateWindowWithSwapchain(&m_window, vk, name.c_str(), width, height) != gfx::Result::SUCCESS) {
            throw std::runtime_error("yo");
        }
    }

    void run() {
        while (true) {
            glfwWaitEvents();
            if (glfwWindowShouldClose(m_window.window)) {
                break;
            }

            //Draw(&app);
        }
        glfwDestroyWindow(m_window.window);
        m_window.window = nullptr;
    }

private:
    std::shared_ptr<Context> m_ctx;
    gfx::Window m_window;
};

namespace nb = nanobind;


typedef ImU32 Color;
// enum WindowFlags {
//     NONE = 0x0,
//     NO_TITLE_BAR = 0x1,
// };

// nb::tuple ImGui_begin(std::string name, std::optional<bool> open, WindowFlags window_flags) {
nb::tuple ImGui_begin(const char* name, std::optional<bool> open, ImGuiWindowFlags_ window_flags) {
    //..
    printf("%s", name);
    return nb::make_tuple(true, false);
}

ImGuiStyle& ImGui_get_style() {
    ImGui::CreateContext();
    return ImGui::GetStyle();
}

NB_MODULE(pyxpg, m) {
    nb::class_<Context>(m, "Context")
        .def(nb::init<>())
    ;

    nb::class_<Window>(m, "Window")
        .def(nb::init<std::shared_ptr<Context>, const::std::string&, u32, u32>())
        .def("run", &Window::run)
    ;

    nb::module_ mod_imgui = m.def_submodule("imgui", "ImGui bindings for XPG");
    #include "generated_imgui.inc"

    // TODO: missing likely more
    nb::class_<ImVec2>(mod_imgui, "Vec2")
        .def_rw("x", &ImVec2::x)
        .def_rw("y", &ImVec2::y);
    nb::class_<ImVec4>(mod_imgui, "Vec4")
        .def_rw("x", &ImVec4::x)
        .def_rw("y", &ImVec4::y)
        .def_rw("z", &ImVec4::z)
        .def_rw("w", &ImVec4::w);

    // Examples:
    // nb::class_<ImGuiStyle>(mod_imgui, "Style")
    //     .def_rw("alpha", &ImGuiStyle::Alpha)
    //     .def_rw("window_padding", &ImGuiStyle::WindowPadding);

    // nb::enum_<WindowFlags>(m2, "WindowFlags")
    //     .value("NONE", WindowFlags::NONE)
    //     .value("NO_TITLE_BAR", WindowFlags::NO_TITLE_BAR)
    //     .export_values();

    // mod_imgui.def("begin", ImGui_begin);
    // mod_imgui.def("get_style", ImGui_get_style, nb::rv_policy::reference);



    // $!
    // !$


}