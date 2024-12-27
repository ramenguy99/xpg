#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/shared_ptr.h>

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

NB_MODULE(xpg, m) {
    nb::class_<Context>(m, "Context")
        .def(nb::init<>())
    ;

    nb::class_<Window>(m, "Window")
        .def(nb::init<std::shared_ptr<Context>, const::std::string&, u32, u32>())
        .def("run", &Window::run)
    ;
}