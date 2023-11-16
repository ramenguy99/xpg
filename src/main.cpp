#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utility> // std::move

#define VOLK_IMPLEMENTATION
#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>

#define _GLFW_VULKAN_STATIC
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
    //OutputDebugStringA(message);
    
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

internal void
Callback_WindowResize(GLFWwindow* window, int width, int height) {
    VulkanWindow* w = (VulkanWindow*)glfwGetWindowUserPointer(window);
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

    // Install callbacks.
    glfwSetWindowSizeCallback(window, Callback_WindowResize);

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
        if (formats[i].format == VK_FORMAT_B8G8R8_UNORM || formats[i].format == VK_FORMAT_R8G8B8_UNORM) {
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

    glfwSetWindowUserPointer(window, w);

    w->window = window;
    w->surface = surface;
    w->swapchain_format = format;

    return VulkanResult::SUCCESS;
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

    VulkanContext vk = {};
    if (InitializeVulkan(&vk, VK_API_VERSION_1_3, instance_extensions, device_extensions, true, true, true, true) != VulkanResult::SUCCESS) {
        printf("Failed to initialize vulkan\n");
        exit(1);
    }

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

    // Initialize queue and command allocator.
    VkResult result;

    VkQueue queue;
    vkGetDeviceQueue(vk.device, vk.queue_family_index, 0, &queue);
    
    Array<VkCommandPool> command_pools(window.images.length);
    Array<VkCommandBuffer> command_buffers(window.images.length);
    for (usize i = 0; i < window.images.length; i++) {
        VkCommandPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;// | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = vk.queue_family_index;
    
        result = vkCreateCommandPool(vk.device, &pool_info, 0, &command_pools[i]);
        assert(result == VK_SUCCESS);

        VkCommandBufferAllocateInfo allocate_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        allocate_info.commandPool = command_pools[i];
        allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate_info.commandBufferCount = 1;

        result = vkAllocateCommandBuffers(vk.device, &allocate_info, &command_buffers[i]);
        assert(result == VK_SUCCESS);
    }
    
    VkCommandBufferBeginInfo begin_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    Array<VkSemaphore> acquire_semaphores(window.images.length);
    Array<VkSemaphore> release_semaphores(window.images.length);
    Array<VkFence> fences(window.images.length);
    for (usize i = 0; i < window.images.length; i++) {
        VkFenceCreateInfo fence_info = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(vk.device, &fence_info, 0, &fences[i]);

        CreateVulkanSemaphore(vk.device, &acquire_semaphores[i]);
        CreateVulkanSemaphore(vk.device, &release_semaphores[i]);
    }

    // Create descriptor pool for imgui.
    VkDescriptorPoolSize pool_sizes[] = {
        // { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
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
        VkCommandPool command_pool = command_pools[0];
        VkCommandBuffer command_buffer= command_buffers[0];

        // Reset command buffer.
        VkResult vkr = vkResetCommandPool(vk.device, command_pool, 0);
        assert(vkr == VK_SUCCESS);

        // Begin recording commands.
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



    bool wait_for_events = false;
    bool force_swapchain_update = false;
    u32 frame_index = 0;

    while (true) {
        if (wait_for_events) {
            glfwWaitEvents();
        }
        else {
            glfwPollEvents();
        }

        if (glfwWindowShouldClose(window.window)) {
            break;
        }

        SwapchainStatus swapchain_status = UpdateSwapchain(&window, vk, force_swapchain_update);
        if (swapchain_status == SwapchainStatus::FAILED) {
            printf("Swapchain update failed\n");
            exit(1);
        }
        force_swapchain_update = false;

        if (swapchain_status == SwapchainStatus::MINIMIZED) {
            wait_for_events = true;
            continue;
        } 
        else if(swapchain_status == SwapchainStatus::RESIZED) {
            // Resize framebuffer sized elements.
        }

        wait_for_events = false;
        
        // Acquire current frame
        VkFence fence = fences[frame_index];
        vkWaitForFences(vk.device, 1, &fence, VK_TRUE, ~0);

        VkSemaphore acquire_semaphore = acquire_semaphores[frame_index];
        VkSemaphore release_semaphore = release_semaphores[frame_index];

        u32 index;
        VkResult vkr = vkAcquireNextImageKHR(vk.device, window.swapchain, ~0ull, acquire_semaphore, 0, &index);
        if(vkr == VK_ERROR_OUT_OF_DATE_KHR) {
            force_swapchain_update = true;
            continue;
        }
        
        ImGui_ImplGlfw_NewFrame();
        ImGui_ImplVulkan_NewFrame();
        ImGui::NewFrame();

        ImGui::DockSpaceOverViewport(NULL, ImGuiDockNodeFlags_PassthruCentralNode);

        ImGui::ShowDemoWindow();
        
        // Render imgui.
        ImGui::Render();
        
        // Reset command pool
        VkCommandPool command_pool = command_pools[frame_index];
        VkCommandBuffer command_buffer= command_buffers[frame_index];

        vkr = vkResetCommandPool(vk.device, command_pool, 0);
        assert(vkr == VK_SUCCESS);

        // Record commands
        vkr = vkBeginCommandBuffer(command_buffer, &begin_info);
        assert(vkr == VK_SUCCESS);
       
        vkResetFences(vk.device, 1, &fence);

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
        
        vkCmdPipelineBarrier(command_buffer,
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

        vkCmdBeginRenderingKHR(command_buffer, &renderingInfo);
        
        ImDrawData* draw_data = ImGui::GetDrawData();
        ImGui_ImplVulkan_RenderDrawData(draw_data, command_buffer);

        vkCmdEndRenderingKHR(command_buffer);

        barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = 0;
        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 
                             0, 0, 
                             0, 0, 
                             1, &barrier);

        vkr = vkEndCommandBuffer(command_buffer);
        assert(vkr == VK_SUCCESS);
        
        // Submit commands
		VkPipelineStageFlags submit_stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        
        VkSubmitInfo submit_info = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = &acquire_semaphore;
        submit_info.pWaitDstStageMask = &submit_stage_mask;
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &release_semaphore;
        
            
        vkr = vkQueueSubmit(queue, 1, &submit_info, fence);
        assert(vkr == VK_SUCCESS);
        
        // Present
        VkPresentInfoKHR present_info = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &window.swapchain;
        present_info.pImageIndices = &index;
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = &release_semaphore;
        vkr = vkQueuePresentKHR(queue, &present_info);

        if (vkr == VK_ERROR_OUT_OF_DATE_KHR || vkr == VK_SUBOPTIMAL_KHR) {
            force_swapchain_update = true;
        } else if (result != VK_SUCCESS) {
            printf("Failed to submit\n");
            exit(1);
        }
        
        // // Wait
        // vkr = vkDeviceWaitIdle(vk.device);
        // assert(vkr == VK_SUCCESS);
        frame_index = (frame_index + 1) % window.images.length;
    }

    // Wait
    vkDeviceWaitIdle(vk.device);
    
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();


    for (usize i = 0; i < window.image_views.length; i++) {
        vkDestroyFence(vk.device, fences[i], 0);

        vkDestroySemaphore(vk.device, acquire_semaphores[i], 0);
        vkDestroySemaphore(vk.device, release_semaphores[i], 0);

        vkFreeCommandBuffers(vk.device, command_pools[i], 1, &command_buffers[i]);
        vkDestroyCommandPool(vk.device, command_pools[i], 0);
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

    system("pause");
}
