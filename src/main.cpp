#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utility> // std::move

#define VOLK_IMPLEMENTATION
#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>

#define VMA_STATIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#define _GLFW_VULKAN_STATIC
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#define XPG_VERSION 0

#undef APIENTRY
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui.h>
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

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t   s8;
typedef int16_t  s16;
typedef int32_t  s32;
typedef int64_t  s64;
typedef size_t   usize;
typedef float    f32;
typedef double   f64;
#define ArrayCount(a) (sizeof((a)) / sizeof(*(a)))
#define ZeroAlloc(size) (calloc(1, size))
#define Free(p) free((p))
#define OutOfBounds(i) (assert(false))
#define OutOfSpace() (assert(false))
#define internal static


template<typename T>
T Max(const T& a, const T& b) {
    return a < b ? b : a;
}

template<typename T>
T Min(const T& a, const T& b) {
    return a < b ? a : b;
}

template<typename T>
T Clamp(const T& v, const T& min, const T& max) {
    return Min(Max(v, min), max);
}

#include "array.h"

static VkBool32 VKAPI_CALL 
VulkanDebugReportCallback(VkDebugReportFlagsEXT flags, 
                           VkDebugReportObjectTypeEXT objectType, 
                           uint64_t object, size_t location, int32_t messageCode, 
                           const char* pLayerPrefix, const char* pMessage, void* pUserData)
{
    // This silences warnings like "For optimal performance image layout should be VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL instead of GENERAL."
    // We'll assume other performance warnings are also not useful.
    //if (flags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT)
    //	return VK_FALSE;
    
    const char* type =
    (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT)
        ? "ERROR"
        : (flags & VK_DEBUG_REPORT_WARNING_BIT_EXT)
        ? "WARNING"
        : "INFO";
    
    char message[4096];
    snprintf(message, sizeof(message), "%s: %s\n", type, pMessage);
    
    
    printf("%s", message);
    OutputDebugStringA(message);
    
    // if (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT)
    //  assert(!"Validation error encountered!");
    
    return VK_FALSE;
}

struct VulkanContext {
    u32 version;
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    u32 queue_family_index;

    // Debug
    VkDebugReportCallbackEXT debug_callback;
};

enum class VulkanResult {
    SUCCESS,

    API_ERROR,
    API_OUT_OF_MEMORY,
    INVALID_VERSION,
    LAYER_NOT_PRESENT,
    EXTENSION_NOT_PRESENT,
    NO_VALID_DEVICE_FOUND,
    DEVICE_CREATION_FAILED,
    SWAPCHAIN_CREATION_FAILED,
    SURFACE_CREATION_FAILED,
};


// internal const char* 
// PhysicalDeviceTypeToString(VkPhysicalDeviceType type) {
//     const char* physical_device_types[] = {
//         "VK_PHYSICAL_DEVICE_TYPE_OTHER",
//         "VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU",
//         "VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
//         "VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU",
//         "VK_PHYSICAL_DEVICE_TYPE_CPU",
//     };
//     
//     if(type < ArrayCount(physical_device_types)) {
//         return physical_device_types[type];
//     } else {
//         return "<unknown device type>";
//     }
// }

VulkanResult 
InitializeVulkan(VulkanContext* vk, u32 required_version, ArrayView<const char*> instance_extensions, ArrayView<const char*> device_extensions, bool require_presentation_support, bool dynamic_rendering, bool enable_validation_layer, bool verbose) {
    VkResult result;

    // Initialize vulkan loader.
    volkInitialize();

    // Query vulkan version.
    u32 version;
    result = vkEnumerateInstanceVersion(&version);
    if (result != VK_SUCCESS) {
        return VulkanResult::API_OUT_OF_MEMORY;
    }

    u32 version_major = VK_API_VERSION_MAJOR(version);
    u32 version_minor = VK_API_VERSION_MINOR(version);
    u32 version_patch = VK_API_VERSION_PATCH(version);
    if (verbose) {
        printf("Vulkan API version %u.%u.%u\n", version_major, version_minor, version_patch);
    }

    // Check if required version is met.
    if (version < required_version) {
        return VulkanResult::INVALID_VERSION;
    }
    
    // Enumerate layer properties.
    u32 layer_properties_count = 0;
    result = vkEnumerateInstanceLayerProperties(&layer_properties_count, NULL);
    if (result != VK_SUCCESS) {
        return VulkanResult::API_OUT_OF_MEMORY;
    }

    Array<VkLayerProperties> layer_properties(layer_properties_count);
    result = vkEnumerateInstanceLayerProperties(&layer_properties_count, layer_properties.data);
    if (result != VK_SUCCESS) {
        return VulkanResult::API_OUT_OF_MEMORY;
    }

    // Check if validation layer is present.
    const char* validation_layer_name = "VK_LAYER_KHRONOS_validation";
    bool validation_layer_present = false;
    for (u32 i = 0; i < layer_properties_count; i++) {
        VkLayerProperties& l = layer_properties[i];
        if (strcmp(l.layerName, validation_layer_name) == 0) {
            validation_layer_present = true;
            break;
        }
    }
    
    // Application info.
    VkApplicationInfo application_info = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    application_info.apiVersion = required_version;
    application_info.pApplicationName = "XPG";
    application_info.applicationVersion = XPG_VERSION;
    
    // Instance info.
    VkInstanceCreateInfo info = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    info.enabledExtensionCount = (u32)instance_extensions.length;
    info.ppEnabledExtensionNames = instance_extensions.data;
    info.pApplicationInfo = &application_info;
    
    const char* enabled_layers[1] = { validation_layer_name };

    if (validation_layer_present) {
        info.enabledLayerCount = ArrayCount(enabled_layers);
        info.ppEnabledLayerNames = enabled_layers;
    }
    
    // Create instance.
    VkInstance instance = 0;
    result = vkCreateInstance(&info, 0, &instance);
    if (result != VK_SUCCESS) {
        switch (result) {
        case VK_ERROR_OUT_OF_HOST_MEMORY:
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:
            return VulkanResult::API_OUT_OF_MEMORY;
        case VK_ERROR_LAYER_NOT_PRESENT:
            return VulkanResult::LAYER_NOT_PRESENT;
        case VK_ERROR_EXTENSION_NOT_PRESENT:
            return VulkanResult::EXTENSION_NOT_PRESENT;
        default:
            return VulkanResult::API_ERROR;
        }
    }
    
    // Load vulkan functions.
    volkLoadInstance(instance);

    
    // Install debug callback.
    VkDebugReportCallbackEXT debug_callback = 0;
    if (validation_layer_present) {
        VkDebugReportCallbackCreateInfoEXT createInfo = { VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT };
        createInfo.flags = VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT | VK_DEBUG_REPORT_ERROR_BIT_EXT;
        createInfo.pfnCallback = VulkanDebugReportCallback;

        result = vkCreateDebugReportCallbackEXT(instance, &createInfo, 0, &debug_callback);
        if (result != VK_SUCCESS) {
            // Failed to install debug callback.
        }
    }
    
    

    // Enumerate and choose a physical devices.
    u32 physical_device_count = 0;
    result = vkEnumeratePhysicalDevices(instance, &physical_device_count, 0);
    if (result != VK_SUCCESS) {
        return VulkanResult::API_OUT_OF_MEMORY;
    }

    Array<VkPhysicalDevice> physical_devices(physical_device_count);
    result = vkEnumeratePhysicalDevices(instance, &physical_device_count, physical_devices.data);
    if (result != VK_SUCCESS) {
        return VulkanResult::API_OUT_OF_MEMORY;
    }

    VkPhysicalDevice physical_device = 0;
    for(u32 i = 0; i < physical_device_count; i++) {
        VkPhysicalDeviceProperties p = {};
        vkGetPhysicalDeviceProperties(physical_devices[i], &p);
        //@Feature: vkGetPhysicalDeviceProperties2, support more / vendor specific device information
        
        bool picked = false;
        if(!physical_device && p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physical_device = physical_devices[i];
            picked = true;
        }

        if (verbose) {
            printf("Physical device %u:%s\n    Name: %s\n    Vulkan version: %d.%d.%d\n    Drivers version: %u\n    Vendor ID: %u\n    Device ID: %u\n    Device type: %s\n",
                i, picked ? " (PICKED)" : "", p.deviceName,
                VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion), VK_API_VERSION_PATCH(p.apiVersion),
                p.driverVersion, p.vendorID, p.deviceID,
                string_VkPhysicalDeviceType(p.deviceType));
                // PhysicalDeviceTypeToString(p.deviceType));
        }
        
#if 0
        // @Incomplete queue and device group information can be used to choose the appropriate device to use
        u32 queue_family_property_count;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_property_count, 0);
        VkQueueFamilyProperties* queue_family_properties = (VkQueueFamilyProperties*)ZeroAlloc(sizeof(VkQueueFamilyProperties) * queue_family_property_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_property_count, queue_family_properties);
        // @Hack assert that there is at least a queue and that it has all the bits we are interested in
        // VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT 
        assert(queue_family_property_count > 0 && queue_family_property_count & 0x7);
        
        
        
        for(u32 j = 0; j < queue_family_property_count; j++) {
            VkQueueFamilyProperties& prop = queue_family_properties[j];
            //@Incomplete vkGetQueueFamilyProperties* versions, support more / vendor specific queue family information
            //printf("Queue: %x - %d\n", prop.queueFlags, prop.queueCount);
            
        }
        
        
        /*
        //@Incomplete group of physical devices with same extensions, see 5.2 of the spec
        u32 count = 1;
        VkPhysicalDeviceGroupProperties group_properties[1];
        vkEnumeratePhysicalDeviceGroups(instance, &count, group_properties);
        */
#endif
    }
    
    // Check that a valid device is found.
    if(!physical_device) {
        return VulkanResult::NO_VALID_DEVICE_FOUND;
    }


    // Create a physical device.
    VkDevice device = 0;

    // @TODO: Queue family index should have been choosen before when picking the device.
    float queue_priorities[] = {1.0f};
    VkDeviceQueueCreateInfo queue_create_info = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    queue_create_info.queueFamilyIndex = 0;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = queue_priorities;
    
    VkDeviceCreateInfo device_create_info = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    device_create_info.queueCreateInfoCount = 1;
    device_create_info.pQueueCreateInfos = &queue_create_info;
    device_create_info.enabledExtensionCount = (u32)device_extensions.length;
    device_create_info.ppEnabledExtensionNames = device_extensions.data;
    device_create_info.pEnabledFeatures = NULL;

    // Enabled dynamic rendering if requested.
    VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamic_rendering_feature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR };
    dynamic_rendering_feature.dynamicRendering = VK_TRUE;
    if (dynamic_rendering) {
        device_create_info.pNext = &dynamic_rendering_feature;
    }

    result = vkCreateDevice(physical_device, &device_create_info, 0, &device);
    if (result != VK_SUCCESS) {
        return VulkanResult::DEVICE_CREATION_FAILED;
    }

    vk->version = version;
    vk->instance = instance;
    vk->physical_device = physical_device;
    vk->device = device;
    vk->queue_family_index = 0;
    vk->debug_callback = debug_callback;

    return VulkanResult::SUCCESS;
}

struct VulkanWindow {
    GLFWwindow* window;
    VkSurfaceKHR surface;
    VkFormat swapchain_format;

    // Swapchain.
    u32 fb_width;
    u32 fb_height;
    VkSwapchainKHR swapchain;
    Array<VkImage> images;
    Array<VkImageView> image_views;
};

VulkanResult CreateVulkanSemaphore(VkDevice device, VkSemaphore* semaphore) {
    VkSemaphoreCreateInfo semaphore_info = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
	VkResult result = vkCreateSemaphore(device, &semaphore_info, 0, semaphore);
    if (result != VK_SUCCESS) {
        return VulkanResult::API_OUT_OF_MEMORY;
    }
	return VulkanResult::SUCCESS;
}

VulkanResult
CreateVulkanSwapchain(VulkanWindow* w, const VulkanContext& vk, VkSurfaceKHR surface, VkFormat format, u32 fb_width, u32 fb_height, usize frames, VkSwapchainKHR old_swapchain) {
    // Create swapchain.
    u32 queue_family_indices[] = { vk.queue_family_index };
    VkSwapchainCreateInfoKHR swapchain_info = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    swapchain_info.surface = surface;
    swapchain_info.minImageCount = (u32)frames;
    swapchain_info.imageFormat = format;
    swapchain_info.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    swapchain_info.imageExtent.width = fb_width;
    swapchain_info.imageExtent.height = fb_height;
    swapchain_info.imageArrayLayers = 1;
    swapchain_info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    swapchain_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchain_info.queueFamilyIndexCount = 1;
    swapchain_info.pQueueFamilyIndices = queue_family_indices;
    swapchain_info.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    swapchain_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchain_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    swapchain_info.oldSwapchain = old_swapchain;

    VkSwapchainKHR swapchain;
    VkResult result = vkCreateSwapchainKHR(vk.device, &swapchain_info, 0, &swapchain);
    if (result != VK_SUCCESS) {
        return VulkanResult::SWAPCHAIN_CREATION_FAILED;
    }

    // Get swapchain images.
    u32 image_count;
    result = vkGetSwapchainImagesKHR(vk.device, swapchain, &image_count, 0);
    if (result != VK_SUCCESS) {
        return VulkanResult::API_OUT_OF_MEMORY;
    }

    Array<VkImage> images(image_count);
    result = vkGetSwapchainImagesKHR(vk.device, swapchain, &image_count, images.data);
    if (result != VK_SUCCESS) {
        return VulkanResult::API_OUT_OF_MEMORY;
    }

    Array<VkImageView> image_views(image_count);
    for (usize i = 0; i < images.length; i++) {
        VkImageViewCreateInfo create_info{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        create_info.image = images[i];
        create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        create_info.format = format;
        create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        create_info.subresourceRange.baseMipLevel = 0;
        create_info.subresourceRange.levelCount = 1;
        create_info.subresourceRange.baseArrayLayer = 0;
        create_info.subresourceRange.layerCount = 1;

        result = vkCreateImageView(vk.device, &create_info, 0, &image_views[i]);
        if (result != VK_SUCCESS) {
            return VulkanResult::API_OUT_OF_MEMORY;
        }
    }

    w->swapchain = swapchain;
    w->images = std::move(images);
    w->image_views = std::move(image_views);
    w->fb_width = fb_width;
    w->fb_height = fb_height;

    return VulkanResult::SUCCESS;
}

enum class SwapchainStatus {
    READY,
    RESIZED,
    MINIMIZED,
    FAILED,
};

SwapchainStatus UpdateSwapchain(VulkanWindow* w, const VulkanContext& vk, bool verbose) {
    VkSurfaceCapabilitiesKHR surface_capabilities;
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.physical_device, w->surface, &surface_capabilities) != VK_SUCCESS) {
        return SwapchainStatus::FAILED;
    }

	uint32_t new_width = surface_capabilities.currentExtent.width;
	uint32_t new_height = surface_capabilities.currentExtent.height;

    if (new_width == 0 || new_height == 0)
        return SwapchainStatus::MINIMIZED;

    if (new_width == w->fb_width && new_height == w->fb_height) {
        return SwapchainStatus::READY;
    }

    for (size_t i = 0; i < w->image_views.length; i++) {
        vkDestroyImageView(vk.device, w->image_views[i], nullptr);
        w->image_views[i] = VK_NULL_HANDLE;
    }

    VkSwapchainKHR old_swapchain = w->swapchain;
    VulkanResult result = CreateVulkanSwapchain(w, vk, w->surface, w->swapchain_format, new_width, new_height, w->images.length, old_swapchain);
    if (result != VulkanResult::SUCCESS) {
        return SwapchainStatus::FAILED;
    }

    vkDeviceWaitIdle(vk.device);

    vkDestroySwapchainKHR(vk.device, old_swapchain, nullptr);

    return SwapchainStatus::RESIZED;
}

VulkanResult 
CreateVulkanWindow(VulkanWindow* w, const VulkanContext& vk, const char* name, u32 width, u32 height, bool verbose) {
    // Create window.
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(width, height, name, NULL, NULL);
    if (!window) {
        printf("Failed to create window\n");
        exit(1);
    }

    // Create window surface.
    VkSurfaceKHR surface = 0;
    VkResult result = glfwCreateWindowSurface(vk.instance, window, NULL, &surface);
    if (result != VK_SUCCESS) {
        return VulkanResult::SURFACE_CREATION_FAILED;
    }

    // Retrieve surface capabilities.
    VkSurfaceCapabilitiesKHR surface_capabilities = {};
    result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.physical_device, surface, &surface_capabilities);
    if (result != VK_SUCCESS) {
        return VulkanResult::SWAPCHAIN_CREATION_FAILED;
    }
    
    // Compute number of frames in flight.
    u32 frames = Max<u32>(2, surface_capabilities.minImageCount);
    if(surface_capabilities.maxImageCount > 0) {
        frames = Min<u32>(frames, surface_capabilities.maxImageCount);
    }

    // Retrieve supported surface formats.
    // @TODO: smarter format picking logic (HDR / non sRGB displays).
    u32 formats_count;
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(vk.physical_device, surface, &formats_count, 0);
    if (result != VK_SUCCESS || formats_count == 0) {
        return VulkanResult::SWAPCHAIN_CREATION_FAILED;
    }

    Array<VkSurfaceFormatKHR> formats(formats_count);
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(vk.physical_device, surface, &formats_count, formats.data);
    if (result != VK_SUCCESS || formats_count == 0) {
        return VulkanResult::SWAPCHAIN_CREATION_FAILED;
    }

    VkFormat format = VK_FORMAT_UNDEFINED;
    for (u32 i = 0; i < formats_count; i++) {
        if (formats[i].format == VK_FORMAT_B8G8R8A8_UNORM || formats[i].format == VK_FORMAT_R8G8B8A8_UNORM) {
            format = formats[i].format;
        }
    }
    
    // If we didn't find what we wanted fall back to the first choice.
    if (format == VK_FORMAT_UNDEFINED) {
        format = formats[0].format;
    }

    if (verbose) {
        printf("Swapchain format: %s\n", string_VkFormat(format));
    }

    // Retrieve framebuffer size.
	u32 fb_width = surface_capabilities.currentExtent.width;
	u32 fb_height = surface_capabilities.currentExtent.height;

    VulkanResult res = CreateVulkanSwapchain(w, vk, surface, format, fb_width, fb_height, frames, VK_NULL_HANDLE);
    if (res != VulkanResult::SUCCESS) {
        return res;
    }

    w->window = window;
    w->surface = surface;
    w->swapchain_format = format;

    return VulkanResult::SUCCESS;
}

struct VulkanFrame {

    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;

    VkSemaphore acquire_semaphore;
    VkSemaphore release_semaphore;

    VkFence fence;
};

struct App {
    VulkanContext* vk;
    VulkanWindow* window;

    VkQueue queue;

    // Swapchain frames, index wraps around at the number of frames in flight.
    u32 frame_index;
    Array<VulkanFrame> frames;

    bool force_swapchain_update;
    bool wait_for_events;

    // Application data.
    u64 current_frame;
    u64 last_frame_timestamp;
    ArrayFixed<f32, 64> frame_times;

    // Rendering
    VkBuffer vertex_buffer;
    VkBuffer index_buffer;
    VkPipeline pipeline;
};

void Draw(App* app) {
    auto& vk = *app->vk;
    auto& window = *app->window;

    u64 timestamp = glfwGetTimerValue();
    
    float dt = (float)((double)(timestamp - app->last_frame_timestamp) / (double)glfwGetTimerFrequency());
    app->last_frame_timestamp = timestamp;
    if (isnan(dt) || isinf(dt)) {
        dt = 0.0f;
    }
    app->frame_times[app->current_frame % app->frame_times.length] = dt;


    float avg_frame_time = 0.0f;
    for (usize i = 0; i < app->frame_times.length; i++) {
        avg_frame_time += app->frame_times[i];
    }
    avg_frame_time /= (f32)app->frame_times.length;

    SwapchainStatus swapchain_status = UpdateSwapchain(&window, vk, app->force_swapchain_update);
    if (swapchain_status == SwapchainStatus::FAILED) {
        printf("Swapchain update failed\n");
        exit(1);
    }
    app->force_swapchain_update = false;

    if (swapchain_status == SwapchainStatus::MINIMIZED) {
        app->wait_for_events = true;
        return;
    } 
    else if(swapchain_status == SwapchainStatus::RESIZED) {
        // Resize framebuffer sized elements.
    }

    app->wait_for_events = false;
    
    // Acquire current frame
    VulkanFrame& frame = app->frames[app->frame_index];

    vkWaitForFences(vk.device, 1, &frame.fence, VK_TRUE, ~0);

    u32 index;
    VkResult vkr = vkAcquireNextImageKHR(vk.device, window.swapchain, ~0ull, frame.acquire_semaphore, 0, &index);
    if(vkr == VK_ERROR_OUT_OF_DATE_KHR) {
        app->force_swapchain_update = true;
        return;
    }
    
    ImGui_ImplGlfw_NewFrame();
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();

    ImGui::DockSpaceOverViewport(NULL, ImGuiDockNodeFlags_PassthruCentralNode);

    if (ImGui::Begin("Playback")) {
        struct Getter {
            static float fn(void* data, int index) {
                App* a = (App*)data;
                
                usize i = (index - (a->current_frame % a->frame_times.length) + a->frame_times.length) % a->frame_times.length;
                return 1.0f / a->frame_times[a->frame_times.length - i - 1];
            }
        };
        //ImGui::PlotLines("", app->frame_times.data, (int)app->frame_times.length, 0, 0, 0.0f, .0f, ImVec2(100, 30));
        ImGui::PlotLines("", Getter::fn, app, (int)app->frame_times.length, 0, 0, 0.0f, 200.0f, ImVec2(100, 30));
        ImGui::SameLine();
        ImGui::Text("FPS: %.2f (%.2fms) [%.2f (%.2fms)]", 1.0f / dt, dt * 1.0e3f, 1.0 / avg_frame_time, avg_frame_time * 1.0e3f);
    }
    ImGui::End();
    //ImGui::ShowDemoWindow();
    
    // Render imgui.
    ImGui::Render();
    
    // Reset command pool
    vkr = vkResetCommandPool(vk.device, frame.command_pool, 0);
    assert(vkr == VK_SUCCESS);

    // Record commands
    VkCommandBufferBeginInfo begin_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkr = vkBeginCommandBuffer(frame.command_buffer, &begin_info);
    assert(vkr == VK_SUCCESS);
   
    vkResetFences(vk.device, 1, &frame.fence);

    VkClearColorValue color = { 0.1f, 0.2f, 0.4f, 1.0f };

    VkImageMemoryBarrier barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = window.images[index];
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    
    barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    
    vkCmdPipelineBarrier(frame.command_buffer,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 
                         0, 0, 
                         0, 0, 
                         1, &barrier);

    // Begin rendering.
    VkRenderingAttachmentInfo attachmentInfo = {};
    attachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    attachmentInfo.imageView = window.image_views[index];
    attachmentInfo.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachmentInfo.resolveMode = VK_RESOLVE_MODE_NONE;
    attachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachmentInfo.clearValue.color = color;

    VkRenderingInfoKHR renderingInfo = {};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
    renderingInfo.renderArea.extent.width = window.fb_width;
    renderingInfo.renderArea.extent.height = window.fb_height;
    renderingInfo.layerCount = 1;
    renderingInfo.viewMask = 0;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &attachmentInfo;

    vkCmdBeginRenderingKHR(frame.command_buffer, &renderingInfo);

    vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app->pipeline);
    
    VkDeviceSize offsets[1] = { 0 };
    vkCmdBindVertexBuffers(frame.command_buffer, 0, 1, &app->vertex_buffer, offsets);

    vkCmdBindIndexBuffer(frame.command_buffer, app->index_buffer, 0, VK_INDEX_TYPE_UINT32);
    
    VkViewport viewport = {};
    viewport.width = (f32)window.fb_width;
    viewport.height = (f32)window.fb_height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(frame.command_buffer, 0, 1, &viewport);

    VkRect2D scissor = {};
    scissor.extent.width = window.fb_width;
    scissor.extent.height = window.fb_height;
    vkCmdSetScissor(frame.command_buffer, 0, 1, &scissor);

    vkCmdDrawIndexed(frame.command_buffer, 3, 1, 0, 0, 0);
    
    ImDrawData* draw_data = ImGui::GetDrawData();
    ImGui_ImplVulkan_RenderDrawData(draw_data, frame.command_buffer);

    vkCmdEndRenderingKHR(frame.command_buffer);

    barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask = 0;
    vkCmdPipelineBarrier(frame.command_buffer,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 
                         0, 0, 
                         0, 0, 
                         1, &barrier);

    vkr = vkEndCommandBuffer(frame.command_buffer);
    assert(vkr == VK_SUCCESS);
    
    // Submit commands
    VkPipelineStageFlags submit_stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    
    VkSubmitInfo submit_info = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &frame.command_buffer;
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = &frame.acquire_semaphore;
    submit_info.pWaitDstStageMask = &submit_stage_mask;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &frame.release_semaphore;
    
        
    vkr = vkQueueSubmit(app->queue, 1, &submit_info, frame.fence);
    assert(vkr == VK_SUCCESS);
    
    // Present
    VkPresentInfoKHR present_info = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &window.swapchain;
    present_info.pImageIndices = &index;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = &frame.release_semaphore;
    vkr = vkQueuePresentKHR(app->queue, &present_info);

    if (vkr == VK_ERROR_OUT_OF_DATE_KHR || vkr == VK_SUBOPTIMAL_KHR) {
        app->force_swapchain_update = true;
    } else if (vkr != VK_SUCCESS) {
        printf("Failed to submit\n");
        exit(1);
    }
    
    // // Wait
    // vkr = vkDeviceWaitIdle(vk.device);
    // assert(vkr == VK_SUCCESS);
    app->frame_index = (app->frame_index + 1) % window.images.length;
    app->current_frame += 1;
}

internal void
Callback_Key(GLFWwindow* window, int key, int scancode, int action, int mods) {
    App* app = (App*)glfwGetWindowUserPointer(window);
    
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(app->window->window, true);
}

internal void
Callback_WindowResize(GLFWwindow* window, int width, int height) {
    App* app = (App*)glfwGetWindowUserPointer(window);
}

internal void
Callback_WindowRefresh(GLFWwindow* window) {
    App* app = (App*)glfwGetWindowUserPointer(window);
    if (app) {
        Draw(app);
    }
}

DWORD WINAPI thread_proc(void* param) {
    HWND window = (HWND)param;
    while (true) {
        SendMessage(window, WM_PAINT, 0, 0);
    }
    return 0;
}


int main(int argc, char** argv) {
    glm::vec3 v = glm::vec3(0.0f);

    // Initialize glfw.
    glfwInit();

    // Check if device supports vulkan.
    if (!glfwVulkanSupported()) {
        printf("Vulkan not found!\n");
        exit(1);
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

    VulkanContext vk = {};
    if (InitializeVulkan(&vk, vulkan_api_version, instance_extensions, device_extensions, true, true, true, true) != VulkanResult::SUCCESS) {
        printf("Failed to initialize vulkan\n");
        exit(1);
    }

    VmaAllocatorCreateInfo vma_info = {};
    vma_info.flags = 0; // Optionally set here that we externally synchronize.
    vma_info.instance = vk.instance;
    vma_info.physicalDevice = vk.physical_device;
    vma_info.device = vk.device;
    vma_info.vulkanApiVersion = vulkan_api_version;

    VmaAllocator vma;
    vmaCreateAllocator(&vma_info, &vma);

    // Check if queue family supports image presentation.
    if (!glfwGetPhysicalDevicePresentationSupport(vk.instance, vk.physical_device, vk.queue_family_index)) {
        printf("Device does not support image presentation\n");
        exit(1);
    }

    VulkanWindow window = {};
    if (CreateVulkanWindow(&window, vk, "XPG", 1600, 900, true) != VulkanResult::SUCCESS) {
        printf("Failed to create vulkan window\n");
        return 1;
    }

    // Redraw during move / resize
    HWND hwnd = glfwGetWin32Window(window.window);
    HANDLE thread = CreateThread(0, 0, thread_proc, hwnd, 0, 0);
    if (thread) {
        CloseHandle(thread);
    }

    // Initialize queue and command allocator.
    VkResult result;

    VkQueue queue;
    vkGetDeviceQueue(vk.device, vk.queue_family_index, 0, &queue);
    
    Array<VulkanFrame> frames(window.images.length);
    for (usize i = 0; i < frames.length; i++) {
        VulkanFrame& frame = frames[i];

        VkCommandPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;// | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = vk.queue_family_index;
    
        result = vkCreateCommandPool(vk.device, &pool_info, 0, &frame.command_pool);
        assert(result == VK_SUCCESS);

        VkCommandBufferAllocateInfo allocate_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        allocate_info.commandPool = frame.command_pool;
        allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate_info.commandBufferCount = 1;

        result = vkAllocateCommandBuffers(vk.device, &allocate_info, &frame.command_buffer);
        assert(result == VK_SUCCESS);

        VkFenceCreateInfo fence_info = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(vk.device, &fence_info, 0, &frame.fence);

        CreateVulkanSemaphore(vk.device, &frame.acquire_semaphore);
        CreateVulkanSemaphore(vk.device, &frame.release_semaphore);
    }

    // Create descriptor pool for imgui.
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
        // { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
    };

    VkDescriptorPoolCreateInfo descriptor_pool_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    descriptor_pool_info.flags = 0;
    descriptor_pool_info.maxSets = 1;
    descriptor_pool_info.pPoolSizes = pool_sizes;
    descriptor_pool_info.poolSizeCount = ArrayCount(pool_sizes);

    VkDescriptorPool descriptor_pool = 0;
    vkCreateDescriptorPool(vk.device, &descriptor_pool_info, 0, &descriptor_pool);

    // Setup window callbacks
    glfwSetWindowSizeCallback(window.window, Callback_WindowResize);
    glfwSetWindowRefreshCallback(window.window, Callback_WindowRefresh);
    glfwSetKeyCallback(window.window, Callback_Key);

    // Initialize ImGui.
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    if (!ImGui_ImplGlfw_InitForVulkan(window.window, true)) {
        printf("Failed to initialize ImGui\n");
        exit(1);
    }

    // TODO: Format, MSAA
    ImGui_ImplVulkan_InitInfo imgui_vk_init_info = {};
    imgui_vk_init_info.Instance = vk.instance;
    imgui_vk_init_info.PhysicalDevice = vk.physical_device;
    imgui_vk_init_info.Device = vk.device;
    imgui_vk_init_info.QueueFamily = vk.queue_family_index;
    imgui_vk_init_info.Queue = queue;
    imgui_vk_init_info.PipelineCache = 0;
    imgui_vk_init_info.DescriptorPool = descriptor_pool;
    imgui_vk_init_info.Subpass = 0;
    imgui_vk_init_info.MinImageCount = (u32)window.images.length;
    imgui_vk_init_info.ImageCount = (u32)window.images.length;
    imgui_vk_init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    imgui_vk_init_info.UseDynamicRendering = true;
    imgui_vk_init_info.ColorAttachmentFormat = window.swapchain_format;
    struct ImGuiCheckResult {
        static void fn(VkResult res) {
            assert(res == VK_SUCCESS);
        }
    };
    imgui_vk_init_info.CheckVkResultFn = ImGuiCheckResult::fn;
    
    if (!ImGui_ImplVulkan_Init(&imgui_vk_init_info, VK_NULL_HANDLE)) {
        printf("Failed to initialize Vulkan imgui backend\n");
        exit(1);
    }

    // Upload font texture.
    {
        VkCommandPool command_pool = frames[0].command_pool;
        VkCommandBuffer command_buffer= frames[0].command_buffer;

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
        // submit_info.waitSemaphoreCount = 0;
        // submit_info.pWaitSemaphores = 0;
        // submit_info.pWaitDstStageMask = 0;
        // submit_info.signalSemaphoreCount = 0;
        // submit_info.pSignalSemaphores = 0;
            
        vkr = vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
        assert(vkr == VK_SUCCESS);

        // Wait for idle.
        vkr = vkDeviceWaitIdle(vk.device);
        assert(vkr == VK_SUCCESS);
    }

    // Allocate vertex and index buffers.
    Array<glm::vec3> vertices;
    vertices.add(glm::vec3(-0.5,  0.5, 0.0));
    vertices.add(glm::vec3( 0.5,  0.5, 0.0));
    vertices.add(glm::vec3( 0.0, -0.5, 0.0));

    Array<u32> indices;
    indices.add(0);
    indices.add(1);
    indices.add(2);

    VkResult vkr;

    VkBufferCreateInfo vertex_buffer_info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    vertex_buffer_info.size = vertices.size_in_bytes();
    vertex_buffer_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
     
    VkBufferCreateInfo index_buffer_info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    index_buffer_info.size = indices.size_in_bytes();
    index_buffer_info.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

    VmaAllocationCreateInfo alloc_create_info = {};
    alloc_create_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    alloc_create_info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    alloc_create_info.flags = 0;
        // VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT  <- Host can access, useful if usage=AUTO, not necessary if require HOST_VISIBLE
        // VMA_ALLOCATION_CREATE_MAPPED_BIT              <- Persistently mapped, not needed for upload once usage as we are doing here, we will manually map and unmap.
     
    VkBuffer vertex_buffer = 0;
    VmaAllocation vertex_allocation = {};
    vkr = vmaCreateBuffer(vma, &vertex_buffer_info, &alloc_create_info, &vertex_buffer, &vertex_allocation, 0);

    VkBuffer index_buffer = 0;
    VmaAllocation index_allocation = {};
    vkr = vmaCreateBuffer(vma, &index_buffer_info, &alloc_create_info, &index_buffer, &index_allocation, 0);

    void* map;

    vmaMapMemory(vma, vertex_allocation, &map);
    ArrayView<u8> vertex_map((u8*)map, vertex_buffer_info.size);
    vertex_map.copy_exact(vertices.as_bytes());
    vmaUnmapMemory(vma, vertex_allocation);

    vmaMapMemory(vma, index_allocation, &map);
    ArrayView<u8> index_map((u8*)map, index_buffer_info.size);
    index_map.copy_exact(indices.as_bytes());
    vmaUnmapMemory(vma, index_allocation);


    // Create graphics pipeline.
    u32 vertex_code[] = {
        // 1112.1.0
        0x07230203,0x00010000,0x0008000b,0x00000018,0x00000000,0x00020011,0x00000001,0x0006000b,
        0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
        0x0007000f,0x00000000,0x00000004,0x6e69616d,0x00000000,0x0000000a,0x0000000f,0x00030003,
        0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,0x00000000,0x00060005,0x00000008,
        0x505f6c67,0x65567265,0x78657472,0x00000000,0x00060006,0x00000008,0x00000000,0x505f6c67,
        0x7469736f,0x006e6f69,0x00030005,0x0000000a,0x00000000,0x00040005,0x0000000f,0x736f5061,
        0x00000000,0x00050048,0x00000008,0x00000000,0x0000000b,0x00000000,0x00030047,0x00000008,
        0x00000002,0x00040047,0x0000000f,0x0000001e,0x00000000,0x00020013,0x00000002,0x00030021,
        0x00000003,0x00000002,0x00030016,0x00000006,0x00000020,0x00040017,0x00000007,0x00000006,
        0x00000004,0x0003001e,0x00000008,0x00000007,0x00040020,0x00000009,0x00000003,0x00000008,
        0x0004003b,0x00000009,0x0000000a,0x00000003,0x00040015,0x0000000b,0x00000020,0x00000001,
        0x0004002b,0x0000000b,0x0000000c,0x00000000,0x00040017,0x0000000d,0x00000006,0x00000003,
        0x00040020,0x0000000e,0x00000001,0x0000000d,0x0004003b,0x0000000e,0x0000000f,0x00000001,
        0x0004002b,0x00000006,0x00000011,0x3f800000,0x00040020,0x00000016,0x00000003,0x00000007,
        0x00050036,0x00000002,0x00000004,0x00000000,0x00000003,0x000200f8,0x00000005,0x0004003d,
        0x0000000d,0x00000010,0x0000000f,0x00050051,0x00000006,0x00000012,0x00000010,0x00000000,
        0x00050051,0x00000006,0x00000013,0x00000010,0x00000001,0x00050051,0x00000006,0x00000014,
        0x00000010,0x00000002,0x00070050,0x00000007,0x00000015,0x00000012,0x00000013,0x00000014,
        0x00000011,0x00050041,0x00000016,0x00000017,0x0000000a,0x0000000c,0x0003003e,0x00000017,
        0x00000015,0x000100fd,0x00010038
    };

    u32 fragment_code[] = {
        // 1112.1.0
        0x07230203,0x00010000,0x0008000b,0x0000000d,0x00000000,0x00020011,0x00000001,0x0006000b,
        0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
        0x0006000f,0x00000004,0x00000004,0x6e69616d,0x00000000,0x00000009,0x00030010,0x00000004,
        0x00000007,0x00030003,0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,0x00000000,
        0x00040005,0x00000009,0x6c6f4366,0x0000726f,0x00040047,0x00000009,0x0000001e,0x00000000,
        0x00020013,0x00000002,0x00030021,0x00000003,0x00000002,0x00030016,0x00000006,0x00000020,
        0x00040017,0x00000007,0x00000006,0x00000004,0x00040020,0x00000008,0x00000003,0x00000007,
        0x0004003b,0x00000008,0x00000009,0x00000003,0x0004002b,0x00000006,0x0000000a,0x3f800000,
        0x0004002b,0x00000006,0x0000000b,0x00000000,0x0007002c,0x00000007,0x0000000c,0x0000000a,
        0x0000000b,0x0000000b,0x0000000a,0x00050036,0x00000002,0x00000004,0x00000000,0x00000003,
        0x000200f8,0x00000005,0x0003003e,0x00000009,0x0000000c,0x000100fd,0x00010038
    };

    VkShaderModuleCreateInfo vertex_module_info = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    vertex_module_info.codeSize = sizeof(vertex_code);
    vertex_module_info.pCode = vertex_code;

    VkShaderModule vertex_module = 0;
    vkr = vkCreateShaderModule(vk.device, &vertex_module_info, 0, &vertex_module);
    assert(vkr == VK_SUCCESS);

    VkShaderModuleCreateInfo fragment_module_info = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    fragment_module_info.codeSize = sizeof(fragment_code);
    fragment_module_info.pCode = fragment_code;
    VkShaderModule fragment_module = 0;
    vkCreateShaderModule(vk.device, &fragment_module_info, 0, &fragment_module);
    assert(vkr == VK_SUCCESS);

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].flags = 0;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT ;
    stages[0].module = vertex_module;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].flags = 0;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT ;
    stages[1].module = fragment_module;
    stages[1].pName = "main";

    VkVertexInputBindingDescription vertex_bindings[1] = {};
    vertex_bindings[0].binding = 0;
    vertex_bindings[0].stride = sizeof(glm::vec3);
    vertex_bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription vertex_attributes[1] = {};
    vertex_attributes[0].location = 0;
    vertex_attributes[0].binding = 0;
    vertex_attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    vertex_attributes[0].offset = 0;

    VkPipelineVertexInputStateCreateInfo vertex_input_state = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    vertex_input_state.vertexBindingDescriptionCount = ArrayCount(vertex_bindings);
    vertex_input_state.pVertexBindingDescriptions = vertex_bindings;
    vertex_input_state.vertexAttributeDescriptionCount = ArrayCount(vertex_attributes);
    vertex_input_state.pVertexAttributeDescriptions = vertex_attributes;

    VkPipelineInputAssemblyStateCreateInfo input_assembly_state = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    input_assembly_state.primitiveRestartEnable = false;
    input_assembly_state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineTessellationStateCreateInfo tessellation_state = { VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO };

    VkPipelineViewportStateCreateInfo viewport_state = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterization_state = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rasterization_state.depthClampEnable = false;
    rasterization_state.rasterizerDiscardEnable = false;
    rasterization_state.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization_state.cullMode = VK_CULL_MODE_NONE;
    rasterization_state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization_state.depthBiasEnable = false;
    rasterization_state.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample_state = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    multisample_state.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth_stencil_state = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    depth_stencil_state.depthTestEnable = false;
    depth_stencil_state.depthWriteEnable = false;
    depth_stencil_state.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState attachments[1] = {};
    attachments[0].blendEnable = false;

    VkPipelineColorBlendStateCreateInfo blend_state = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    blend_state.attachmentCount = ArrayCount(attachments);
    blend_state.pAttachments = attachments;

    VkDynamicState dynamic_states[2] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    VkPipelineDynamicStateCreateInfo dynamic_state = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynamic_state.dynamicStateCount = ArrayCount(dynamic_states);
    dynamic_state.pDynamicStates = dynamic_states;

    // Layout
    VkPipelineLayoutCreateInfo layout_info = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };

    VkPipelineLayout layout = 0;
    vkr = vkCreatePipelineLayout(vk.device, &layout_info, 0, &layout);
    assert(vkr == VK_SUCCESS);


    VkGraphicsPipelineCreateInfo pipeline_create_info = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pipeline_create_info.flags = 0;
    
    // Shaders
    pipeline_create_info.stageCount = 2;
    pipeline_create_info.pStages = stages;

    // Graphics state
    pipeline_create_info.pVertexInputState = &vertex_input_state;
    pipeline_create_info.pInputAssemblyState = &input_assembly_state;
    pipeline_create_info.pTessellationState = &tessellation_state;
    pipeline_create_info.pViewportState = &viewport_state;
    pipeline_create_info.pRasterizationState = &rasterization_state;
    pipeline_create_info.pMultisampleState = &multisample_state;
    pipeline_create_info.pDepthStencilState = &depth_stencil_state;
    pipeline_create_info.pColorBlendState = &blend_state;
    pipeline_create_info.pDynamicState = &dynamic_state;

    // Binding layout
    pipeline_create_info.layout = layout;

    // Render pass -> we use dynamic rendering
    pipeline_create_info.renderPass = VK_NULL_HANDLE;


    VkPipeline pipeline = 0;
    vkr = vkCreateGraphicsPipelines(vk.device, VK_NULL_HANDLE, 1, &pipeline_create_info, 0, &pipeline);
    assert(vkr == VK_SUCCESS);


    App app = {};
    app.frames = std::move(frames);
    app.window = &window;
    app.vk = &vk;
    app.queue = queue;
    app.wait_for_events = true;
    app.frame_times.resize(ArrayCount(app.frame_times.data));
    app.last_frame_timestamp = glfwGetTimerValue();
    app.vertex_buffer = vertex_buffer;
    app.index_buffer = index_buffer;
    app.pipeline = pipeline;

    glfwSetWindowUserPointer(window.window, &app);

    while (true) {
        if (app.wait_for_events) {
            glfwWaitEvents();
        }
        else {
            glfwPollEvents();
        }

        if (glfwWindowShouldClose(window.window)) {
            break;
        }

        Draw(&app);
    }


    // Wait
    vkDeviceWaitIdle(vk.device);
    
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    vmaDestroyBuffer(vma, vertex_buffer, vertex_allocation);
    vmaDestroyBuffer(vma, index_buffer, index_allocation);
    vmaDestroyAllocator(vma);

    vkDestroyShaderModule(vk.device, vertex_module, 0);
    vkDestroyShaderModule(vk.device, fragment_module, 0);
    vkDestroyPipelineLayout(vk.device, layout, 0);
    vkDestroyPipeline(vk.device, pipeline, 0);

    for (usize i = 0; i < window.image_views.length; i++) {
        VulkanFrame& frame = app.frames[i];
        vkDestroyFence(vk.device, frame.fence, 0);

        vkDestroySemaphore(vk.device, frame.acquire_semaphore, 0);
        vkDestroySemaphore(vk.device, frame.release_semaphore, 0);

        vkFreeCommandBuffers(vk.device, frame.command_pool, 1, &frame.command_buffer);
        vkDestroyCommandPool(vk.device, frame.command_pool, 0);
    }
    
    // Window stuff
    vkDestroyDescriptorPool(vk.device, descriptor_pool, 0);
    for (usize i = 0; i < window.image_views.length; i++) {
        vkDestroyImageView(vk.device, window.image_views[i], 0);
    }
    vkDestroySwapchainKHR(vk.device, window.swapchain, 0);
    vkDestroySurfaceKHR(vk.instance, window.surface, 0);

    vkDestroyDevice(vk.device, 0);
	vkDestroyDebugReportCallbackEXT(vk.instance, vk.debug_callback, 0);
    vkDestroyInstance(vk.instance, 0);

    //system("pause");
}
