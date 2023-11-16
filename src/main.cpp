#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOLK_IMPLEMENTATION
#include <volk.h>

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
#define internal static
#include "array.h"


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

    // Swapchain
    VkSwapchainKHR swapchain;
    Array<VkImage> images;
    Array<VkImageView> image_views;
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
};


internal const char* 
PhysicalDeviceTypeToString(VkPhysicalDeviceType type) {
    const char* physical_device_types[] = {
        "VK_PHYSICAL_DEVICE_TYPE_OTHER",
        "VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU",
        "VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
        "VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU",
        "VK_PHYSICAL_DEVICE_TYPE_CPU",
    };
    
    if(type < ArrayCount(physical_device_types)) {
        return physical_device_types[type];
    } else {
        return "<unknown device type>";
    }
}


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
                PhysicalDeviceTypeToString(p.deviceType));
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


VulkanResult InitializeSwapchain(VulkanContext* vk, VkSurfaceKHR surface, VkFormat format, u32 width, u32 height) {
    VkResult result;

    // Retrieve surface capabilities.
    VkSurfaceCapabilitiesKHR surface_capabilities = {};
    result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk->physical_device, surface, &surface_capabilities);
    if (result != VK_SUCCESS) {
        return VulkanResult::SWAPCHAIN_CREATION_FAILED;
    }

    // Create swapchain.
    u32 queue_family_indices[] = { vk->queue_family_index };
    VkSwapchainCreateInfoKHR swapchain_info = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    swapchain_info.surface = surface;
    swapchain_info.minImageCount = Clamp(2u, surface_capabilities.minImageCount, surface_capabilities.maxImageCount);
    swapchain_info.imageFormat = format;
    swapchain_info.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    swapchain_info.imageExtent.width = width;
    swapchain_info.imageExtent.height = height;
    swapchain_info.imageArrayLayers = 1;
    swapchain_info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    swapchain_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchain_info.queueFamilyIndexCount = 1;
    swapchain_info.pQueueFamilyIndices = queue_family_indices;
    swapchain_info.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    swapchain_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchain_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    swapchain_info.oldSwapchain = VK_NULL_HANDLE;

    VkSwapchainKHR swapchain;
    result = vkCreateSwapchainKHR(vk->device, &swapchain_info, 0, &swapchain);
    if (result != VK_SUCCESS) {
        return VulkanResult::SWAPCHAIN_CREATION_FAILED;
    }

    // Get swapchain images.
    u32 image_count;
    result = vkGetSwapchainImagesKHR(vk->device, swapchain, &image_count, 0);
    if (result != VK_SUCCESS) {
        return VulkanResult::API_OUT_OF_MEMORY;
    }

    Array<VkImage> images(image_count);
    result = vkGetSwapchainImagesKHR(vk->device, swapchain, &image_count, images.data);
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

        result = vkCreateImageView(vk->device, &create_info, 0, &image_views[i]);
        if (result != VK_SUCCESS) {
            return VulkanResult::API_OUT_OF_MEMORY;
        }
    }

    vk->swapchain = swapchain;
    vk->images = std::move(images);
    vk->image_views = std::move(image_views);

    return VulkanResult::SUCCESS;
}

VulkanResult CreateVulkanSemaphore(VkDevice device, VkSemaphore* semaphore) {
    VkSemaphoreCreateInfo semaphore_info = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
	VkResult result = vkCreateSemaphore(device, &semaphore_info, 0, semaphore);
    if (result != VK_SUCCESS) {
        return VulkanResult::API_OUT_OF_MEMORY;
    }
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

    // Create window.
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(1600, 900, "XGP", NULL, NULL);
    if (!window) {
        printf("Failed to create window\n");
        exit(1);
    }

    // Create window surface.
    VkSurfaceKHR surface = 0;
    VkResult result = glfwCreateWindowSurface(vk.instance, window, NULL, &surface);
    if (result != VK_SUCCESS) {
        printf("Failed to create surface for window\n");
        exit(1);
    }

    // Initialize swapchain for this surface.
    int fb_width, fb_height;
    glfwGetFramebufferSize(window, &fb_width, &fb_height);
    if (InitializeSwapchain(&vk, surface, VK_FORMAT_B8G8R8A8_UNORM, (u32)fb_width, (u32)fb_height) != VulkanResult::SUCCESS) {
        printf("Failed to initialize swapchain\n");
        exit(1);
    }

    // Initialize queue and command allocator.
    VkQueue queue;
    vkGetDeviceQueue(vk.device, vk.queue_family_index, 0, &queue);
    
    VkCommandPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = vk.queue_family_index;
    
    VkCommandPool pool;
    result = vkCreateCommandPool(vk.device, &pool_info, 0, &pool);
    assert(result == VK_SUCCESS);
    
    VkCommandBufferAllocateInfo allocate_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocate_info.commandPool = pool;
    allocate_info.level =VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate_info.commandBufferCount = 1;
    
    VkCommandBuffer command_buffer;
    result = vkAllocateCommandBuffers(vk.device, &allocate_info, &command_buffer);
    assert(result == VK_SUCCESS);
    
    VkCommandBufferBeginInfo begin_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    VkSemaphore acquire_semaphore = 0, release_semaphore = 0;
    CreateVulkanSemaphore(vk.device, &acquire_semaphore);
    CreateVulkanSemaphore(vk.device, &release_semaphore);

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

    if (!ImGui_ImplGlfw_InitForVulkan(window, true)) {
        printf("Failed to initialize GLFW imgui backend\n");
        exit(1);
    }

    // TODO: frames in flight, format, MSAA
    ImGui_ImplVulkan_InitInfo imgui_vk_init_info = {};
    imgui_vk_init_info.Instance = vk.instance;
    imgui_vk_init_info.PhysicalDevice = vk.physical_device;
    imgui_vk_init_info.Device = vk.device;
    imgui_vk_init_info.QueueFamily = vk.queue_family_index;
    imgui_vk_init_info.Queue = queue;
    imgui_vk_init_info.PipelineCache = 0;
    imgui_vk_init_info.DescriptorPool = descriptor_pool;
    imgui_vk_init_info.Subpass = 0;
    imgui_vk_init_info.MinImageCount = 2;
    imgui_vk_init_info.ImageCount = 2;
    imgui_vk_init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    imgui_vk_init_info.UseDynamicRendering = true;
    imgui_vk_init_info.ColorAttachmentFormat = VK_FORMAT_B8G8R8A8_UNORM;
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
        // Reset command buffer.
        VkResult vkr = vkResetCommandBuffer(command_buffer, 0);
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
    
    bool animating = true;
    while (true) {
        if (animating) {
            glfwPollEvents();
        }
        else {
            glfwWaitEvents();
        }

        if (glfwWindowShouldClose(window)) {
            break;
        }
        
        ImGui_ImplGlfw_NewFrame();
        ImGui_ImplVulkan_NewFrame();
        ImGui::NewFrame();

        ImGui::DockSpaceOverViewport(NULL, ImGuiDockNodeFlags_PassthruCentralNode);

        ImGui::ShowDemoWindow();
        
        // Render imgui.
        ImGui::Render();
        
        // Reset command buffer
        VkResult vkr = vkResetCommandBuffer(command_buffer, 0);
        assert(vkr == VK_SUCCESS);

        // Acquire current frame
        u32 index;
        vkr = vkAcquireNextImageKHR(vk.device, vk.swapchain, ~0ull, acquire_semaphore, 0, &index);
        assert(vkr == VK_SUCCESS);
        
        // Record commands
        vkr = vkBeginCommandBuffer(command_buffer, &begin_info);
        assert(vkr == VK_SUCCESS);
        
		VkClearColorValue color = { 0.1f, 0.2f, 0.4f, 1.0f };

        VkImageMemoryBarrier barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = vk.images[index];
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
        attachmentInfo.imageView = vk.image_views[index];
        attachmentInfo.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        attachmentInfo.resolveMode = VK_RESOLVE_MODE_NONE;
        attachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachmentInfo.clearValue.color = color;

        VkRenderingInfoKHR renderingInfo = {};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
        renderingInfo.renderArea.extent.width = fb_width;
        renderingInfo.renderArea.extent.height = fb_height;
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
        
            
        vkr = vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
        assert(vkr == VK_SUCCESS);
        
        // Present
        VkPresentInfoKHR present_info = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &vk.swapchain;
        present_info.pImageIndices = &index;
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = &release_semaphore;
        vkQueuePresentKHR(queue, &present_info);
        
        // wait
        vkr = vkDeviceWaitIdle(vk.device);
        assert(vkr == VK_SUCCESS);
    }

    // Wait
    vkDeviceWaitIdle(vk.device);
    
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();

    vkDestroySemaphore(vk.device, acquire_semaphore, 0);
    vkDestroySemaphore(vk.device, release_semaphore, 0);
    vkFreeCommandBuffers(vk.device, pool, 1, &command_buffer);
    vkDestroyCommandPool(vk.device, pool, 0);
    vkDestroyDescriptorPool(vk.device, descriptor_pool, 0);
    for (usize i = 0; i < vk.image_views.length; i++) {
        vkDestroyImageView(vk.device, vk.image_views[i], 0);
    }
    vkDestroySwapchainKHR(vk.device, vk.swapchain, 0);
    vkDestroySurfaceKHR(vk.instance, surface, 0);
    vkDestroyDevice(vk.device, 0);
	vkDestroyDebugReportCallbackEXT(vk.instance, vk.debug_callback, 0);
    vkDestroyInstance(vk.instance, 0);

    system("pause");
}