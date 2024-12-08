
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utility> // std::move
#include <functional> // std::function
#include <mutex>
#include <unordered_map>

#ifdef _WIN32
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <semaphore.h>
#include <pthread.h>
#endif

#define VOLK_IMPLEMENTATION
#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>

#define VMA_STATIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#define _GLFW_VULKAN_STATIC
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#endif
#include <GLFW/glfw3.h>


#undef APIENTRY
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui.h>
#include <imgui.cpp>
#include <imgui_demo.cpp>
#include <imgui_draw.cpp>
#include <imgui_tables.cpp>
#include <imgui_widgets.cpp>

#include <implot.h>
#include <implot_internal.h>
#include <implot.cpp>
#include <implot_items.cpp>
#include <implot_demo.cpp>

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_glfw.cpp>

#undef VK_NO_PROTOTYPES
#include <backends/imgui_impl_vulkan.h>
#include <backends/imgui_impl_vulkan.cpp>

#include <atomic_queue/atomic_queue.h>

#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/shared_ptr.h>

#define XPG_VERSION 0

#include "defines.h"
#include "array.h"
#include "platform.h"
#include "threading.h"
#include "gfx.h"
#include "buffered_stream.h"

class Context {
public:
    Context() {
        // Initialize glfw.
        glfwInit();

        // Check if device supports vulkan.
        if (!glfwVulkanSupported()) {
            throw std::runtime_error("vulkan not found");
        }

        // Get instance extensions required by GLFW.
        u32 glfw_instance_extensions_count;
        const char** glfw_instance_extensions = glfwGetRequiredInstanceExtensions(&glfw_instance_extensions_count);

        // Vulkan initialization.
        Array<const char*> instance_extensions;
        for(u32 i = 0; i < glfw_instance_extensions_count; i++) {
            instance_extensions.add(glfw_instance_extensions[i]);
        }
        instance_extensions.add("VK_EXT_debug_report");
        
        Array<const char*> device_extensions;
        device_extensions.add("VK_KHR_swapchain");
        device_extensions.add("VK_KHR_dynamic_rendering");

        u32 vulkan_api_version = VK_API_VERSION_1_3;

        if (gfx::InitializeContext(&m_vk, vulkan_api_version, instance_extensions, device_extensions, true, gfx::DeviceFeatures::DYNAMIC_RENDERING | gfx::DeviceFeatures::DESCRIPTOR_INDEXING, true, true) != gfx::Result::SUCCESS) {
            throw std::runtime_error("vulkan failed to initialize");
        }
        vkGetDeviceQueue(m_vk.device, m_vk.queue_family_index, 0, &m_queue);
    }

    ~Context() {
        // TODO: cleanup
    }

private:
    gfx::Context m_vk;
    VkQueue m_queue;
    
    friend class Window;
};

class Window {
public:
    Window(std::shared_ptr<Context> ctx, const std::string& name, u32 width, u32 height)
        : m_ctx(ctx)
    {
        gfx::Context& vk = ctx->m_vk;

        if (CreateWindowWithSwapchain(&m_window, vk, name.c_str(), width, height, true) != gfx::Result::SUCCESS) {
            throw std::runtime_error("yo");
        }

        m_frames.resize(m_window.images.length);
        for (usize i = 0; i < m_frames.length; i++) {
            gfx::Frame& frame = m_frames[i];

            VkCommandPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
            pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;// | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pool_info.queueFamilyIndex = vk.queue_family_index;
        
            VkResult result = vkCreateCommandPool(vk.device, &pool_info, 0, &frame.command_pool);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("vulkan failed to create command pool");
            }

            VkCommandBufferAllocateInfo allocate_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            allocate_info.commandPool = frame.command_pool;
            allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocate_info.commandBufferCount = 1;

            result = vkAllocateCommandBuffers(vk.device, &allocate_info, &frame.command_buffer);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("vulkan failed to create command buffers");
            }

            VkFenceCreateInfo fence_info = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
            fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            vkCreateFence(vk.device, &fence_info, 0, &frame.fence);

            gfx::CreateGPUSemaphore(vk.device, &frame.acquire_semaphore);
            gfx::CreateGPUSemaphore(vk.device, &frame.release_semaphore);
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
    Array<gfx::Frame> m_frames;
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