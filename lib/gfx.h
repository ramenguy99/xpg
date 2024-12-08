namespace gfx {

struct Frame
{
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;

    VkSemaphore acquire_semaphore;
    VkSemaphore release_semaphore;

    VkFence fence;

    // Filled every frame after acquiring
    VkImage current_image;
    VkImageView current_image_view;
    u32 current_image_index;
};

struct Window
{
    GLFWwindow* window;
    VkSurfaceKHR surface;
    VkFormat swapchain_format;

    // Swapchain
    u32 fb_width;
    u32 fb_height;
    VkSwapchainKHR swapchain;

    // Index in swapchain frames, wraps around at the number of frames in flight.
    // This is always sequential, it should be used to index into 'frames' but not
    // into images and image_views. For that one should use the index returned
    // by vkAcquireNextImageKHR.
    u32 swapchain_frame_index;

    // If set to true when calling UpdateSwapchain it will force creation of a new swapchain.
    // Should be set after a vkQueuePresentKHR returns VK_ERROR_OUT_OF_DATE_KHR or VK_SUBOPTIMAL_KHR.
    bool force_swapchain_recreate;

    // Per frame swapchain data
    Array<VkImage> images;
    Array<VkImageView> image_views;

    // Per frame persistent data
    Array<Frame> frames;
};

struct Context
{
    u32 version;
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    u32 queue_family_index;
    VmaAllocator vma;

    // Debug
    VkDebugReportCallbackEXT debug_callback;
};

struct UniformBuffer
{
    VkBuffer buffer;
    ArrayView<u8> map;
    VmaAllocation allocation;
};

struct DepthBuffer
{
    VkImage image;
    VkImageView view;
    VmaAllocation allocation;
};

enum class Result
{
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
    VMA_CREATION_FAILED,
};

enum class SwapchainStatus
{
    READY,
    RESIZED,
    MINIMIZED,
    FAILED,
};

enum class PresentResult
{
    SUCCESS,
    FAILED,
};

// TODO: substitute this with more extendable logic once
// we rework device picking logic.
struct DeviceFeatures {
    enum DeviceFeaturesFlags {
        NONE = 0,
        DESCRIPTOR_INDEXING = 1,
        DYNAMIC_RENDERING = 2,
    };

    DeviceFeatures(DeviceFeaturesFlags flags): flags(flags) {}
    DeviceFeatures(uint64_t flags): flags((DeviceFeaturesFlags)flags) {}

    operator bool() {
        return (uint64_t)flags != 0;
    }

    DeviceFeatures operator|(const DeviceFeaturesFlags& b) {
        return (DeviceFeaturesFlags)((uint64_t)flags | (uint64_t)b);
    }

    DeviceFeatures operator|(const DeviceFeatures& b) {
        return (DeviceFeaturesFlags)((uint64_t)flags | (uint64_t)b.flags);
    }

    DeviceFeatures operator&(const DeviceFeaturesFlags& b) {
        return (DeviceFeaturesFlags)((uint64_t)flags & (uint64_t)b);
    }

    DeviceFeatures operator&(const DeviceFeatures& b) {
        return (DeviceFeaturesFlags)((uint64_t)flags & (uint64_t)b.flags);
    }

    DeviceFeaturesFlags flags;
};


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
#ifdef _WIN32
    OutputDebugStringA(message);
#endif

    // if (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT)
    //  assert(!"Validation error encountered!");

    return VK_FALSE;
}

Result
CreateContext(Context* vk, u32 required_version, ArrayView<const char*> instance_extensions, ArrayView<const char*> device_extensions, bool require_presentation_support, DeviceFeatures device_features, bool enable_validation_layer, bool verbose)
{
    VkResult result;

    // Initialize vulkan loader.
    volkInitialize();

    // Query vulkan version.
    u32 version;
    result = vkEnumerateInstanceVersion(&version);
    if (result != VK_SUCCESS) {
        return Result::API_OUT_OF_MEMORY;
    }

    u32 version_major = VK_API_VERSION_MAJOR(version);
    u32 version_minor = VK_API_VERSION_MINOR(version);
    u32 version_patch = VK_API_VERSION_PATCH(version);
    if (verbose) {
        printf("Vulkan API version %u.%u.%u\n", version_major, version_minor, version_patch);
    }

    // Check if required version is met.
    if (version < required_version) {
        return Result::INVALID_VERSION;
    }

    // Enumerate layer properties.
    u32 layer_properties_count = 0;
    result = vkEnumerateInstanceLayerProperties(&layer_properties_count, NULL);
    if (result != VK_SUCCESS) {
        return Result::API_OUT_OF_MEMORY;
    }

    Array<VkLayerProperties> layer_properties(layer_properties_count);
    result = vkEnumerateInstanceLayerProperties(&layer_properties_count, layer_properties.data);
    if (result != VK_SUCCESS) {
        return Result::API_OUT_OF_MEMORY;
    }

    // Check if validation layer is present.
    const char* validation_layer_name = "VK_LAYER_KHRONOS_validation";
    bool validation_layer_present = false;
    for (u32 i = 0; i < layer_properties_count; i++) {
        VkLayerProperties& l = layer_properties[i];
        if (strcmp(l.layerName, validation_layer_name) == 0) {
            validation_layer_present = true;
            if (verbose) {
                printf("Vulkan validation layer found\n");
            }
            break;
        }
    }
    bool use_validation_layer = validation_layer_present && enable_validation_layer;

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

    if (use_validation_layer) {
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
            return Result::API_OUT_OF_MEMORY;
        case VK_ERROR_LAYER_NOT_PRESENT:
            return Result::LAYER_NOT_PRESENT;
        case VK_ERROR_EXTENSION_NOT_PRESENT:
            return Result::EXTENSION_NOT_PRESENT;
        default:
            return Result::API_ERROR;
        }
    }

    // Load vulkan functions.
    volkLoadInstance(instance);


    // Install debug callback.
    VkDebugReportCallbackEXT debug_callback = 0;
    if (use_validation_layer) {
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
        return Result::API_OUT_OF_MEMORY;
    }

    Array<VkPhysicalDevice> physical_devices(physical_device_count);
    result = vkEnumeratePhysicalDevices(instance, &physical_device_count, physical_devices.data);
    if (result != VK_SUCCESS) {
        return Result::API_OUT_OF_MEMORY;
    }

    VkPhysicalDevice physical_device = 0;
    u32 queue_family_index = 0;
    for (u32 i = 0; i < physical_device_count; i++) {
        VkPhysicalDeviceProperties p = {};
        vkGetPhysicalDeviceProperties(physical_devices[i], &p);
        //@Feature: vkGetPhysicalDeviceProperties2, support more / vendor specific device information

        bool picked = false;
        if (!physical_device || p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
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



        for (u32 j = 0; j < queue_family_property_count; j++) {
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
    if (!physical_device) {
        return Result::NO_VALID_DEVICE_FOUND;
    }


    // Create a physical device.
    VkDevice device = 0;

    // @TODO: Queue family index should have been choosen before when picking the device.
    float queue_priorities[] = { 1.0f };
    VkDeviceQueueCreateInfo queue_create_info = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
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

    if (device_features & DeviceFeatures::DYNAMIC_RENDERING) {
        device_create_info.pNext = &dynamic_rendering_feature;
    }

    VkPhysicalDeviceDescriptorIndexingFeatures descriptor_indexing_features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES };
    // descriptor_indexing_features.shaderInputAttachmentArrayDynamicIndexing = VK_TRUE;
    // descriptor_indexing_features.shaderUniformTexelBufferArrayDynamicIndexing = VK_TRUE;
    // descriptor_indexing_features.shaderStorageTexelBufferArrayDynamicIndexing = VK_TRUE;
    descriptor_indexing_features.shaderUniformBufferArrayNonUniformIndexing = VK_TRUE;
    descriptor_indexing_features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    descriptor_indexing_features.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
    descriptor_indexing_features.shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
    // descriptor_indexing_features.shaderInputAttachmentArrayNonUniformIndexing = VK_TRUE;
    // descriptor_indexing_features.shaderUniformTexelBufferArrayNonUniformIndexing = VK_TRUE;
    // descriptor_indexing_features.shaderStorageTexelBufferArrayNonUniformIndexing = VK_TRUE;
    descriptor_indexing_features.descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE;
    descriptor_indexing_features.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    descriptor_indexing_features.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
    descriptor_indexing_features.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
    // descriptor_indexing_features.descriptorBindingUniformTexelBufferUpdateAfterBind = VK_TRUE;
    // descriptor_indexing_features.descriptorBindingStorageTexelBufferUpdateAfterBind = VK_TRUE;
    // descriptor_indexing_features.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
    descriptor_indexing_features.descriptorBindingPartiallyBound = VK_TRUE;
    // descriptor_indexing_features.descriptorBindingVariableDescriptorCount = VK_TRUE;
    descriptor_indexing_features.runtimeDescriptorArray = VK_TRUE;

    if (device_features & DeviceFeatures::DESCRIPTOR_INDEXING) {
        descriptor_indexing_features.pNext = const_cast<void*>(device_create_info.pNext);
        device_create_info.pNext = &descriptor_indexing_features;
    }

    result = vkCreateDevice(physical_device, &device_create_info, 0, &device);
    if (result != VK_SUCCESS) {
        return Result::DEVICE_CREATION_FAILED;
    }

    VmaAllocatorCreateInfo vma_info = {};
    vma_info.flags = 0; // Optionally set here that we externally synchronize.
    vma_info.instance = instance;
    vma_info.physicalDevice = physical_device;
    vma_info.device = device;
    vma_info.vulkanApiVersion = version;

    VmaAllocator vma;
    result = vmaCreateAllocator(&vma_info, &vma);
    if (result != VK_SUCCESS) {
        return Result::VMA_CREATION_FAILED;
    }

    VkQueue queue;
    vkGetDeviceQueue(device, queue_family_index, 0, &queue);

    vk->version = version;
    vk->instance = instance;
    vk->physical_device = physical_device;
    vk->device = device;
    vk->queue = queue;
    vk->queue_family_index = queue_family_index;
    vk->debug_callback = debug_callback;
    vk->vma = vma;

    return Result::SUCCESS;
}

void DestroyContext(Context* vk) {
    // VMA
    vmaDestroyAllocator(vk->vma);

    // Device and instance
    vkDestroyDevice(vk->device, 0);
	vkDestroyDebugReportCallbackEXT(vk->instance, vk->debug_callback, 0);
    vkDestroyInstance(vk->instance, 0);
}

Result
CreateGPUSemaphore(VkDevice device, VkSemaphore* semaphore)
{
    VkSemaphoreCreateInfo semaphore_info = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkResult result = vkCreateSemaphore(device, &semaphore_info, 0, semaphore);
    if (result != VK_SUCCESS) {
        return Result::API_OUT_OF_MEMORY;
    }
    return Result::SUCCESS;
}


Result
CreateSwapchain(Window* w, const Context& vk, VkSurfaceKHR surface, VkFormat format, u32 fb_width, u32 fb_height, usize frames, VkSwapchainKHR old_swapchain)
{
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
    //swapchain_info.presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    swapchain_info.oldSwapchain = old_swapchain;

    VkSwapchainKHR swapchain;
    VkResult result = vkCreateSwapchainKHR(vk.device, &swapchain_info, 0, &swapchain);
    if (result != VK_SUCCESS) {
        return Result::SWAPCHAIN_CREATION_FAILED;
    }

    // Get swapchain images.
    u32 image_count;
    result = vkGetSwapchainImagesKHR(vk.device, swapchain, &image_count, 0);
    if (result != VK_SUCCESS) {
        return Result::API_OUT_OF_MEMORY;
    }

    Array<VkImage> images(image_count);
    result = vkGetSwapchainImagesKHR(vk.device, swapchain, &image_count, images.data);
    if (result != VK_SUCCESS) {
        return Result::API_OUT_OF_MEMORY;
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
            return Result::API_OUT_OF_MEMORY;
        }
    }

    w->swapchain = swapchain;
    w->images = std::move(images);
    w->image_views = std::move(image_views);
    w->fb_width = fb_width;
    w->fb_height = fb_height;

    return Result::SUCCESS;
}

SwapchainStatus UpdateSwapchain(Window* w, const Context& vk)
{
    VkSurfaceCapabilitiesKHR surface_capabilities;
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.physical_device, w->surface, &surface_capabilities) != VK_SUCCESS) {
        return SwapchainStatus::FAILED;
    }

    uint32_t new_width = surface_capabilities.currentExtent.width;
    uint32_t new_height = surface_capabilities.currentExtent.height;

    if (new_width == 0 || new_height == 0)
        return SwapchainStatus::MINIMIZED;

    if (!w->force_swapchain_recreate && (new_width == w->fb_width && new_height == w->fb_height)) {
        return SwapchainStatus::READY;
    }

    vkDeviceWaitIdle(vk.device);

    for (size_t i = 0; i < w->image_views.length; i++) {
        vkDestroyImageView(vk.device, w->image_views[i], nullptr);
        w->image_views[i] = VK_NULL_HANDLE;
    }

    VkSwapchainKHR old_swapchain = w->swapchain;
    Result result = CreateSwapchain(w, vk, w->surface, w->swapchain_format, new_width, new_height, w->images.length, old_swapchain);
    if (result != Result::SUCCESS) {
        return SwapchainStatus::FAILED;
    }

    w->force_swapchain_recreate = false;

    vkDestroySwapchainKHR(vk.device, old_swapchain, nullptr);

    return SwapchainStatus::RESIZED;
}

Frame* AcquireNextFrame(Window* w, const Context& vk) {
    Frame& frame = w->frames[w->swapchain_frame_index];

    // Wait for frame to be done
    vkWaitForFences(vk.device, 1, &frame.fence, VK_TRUE, ~0);

    u32 image_index;
    VkResult vkr = vkAcquireNextImageKHR(vk.device, w->swapchain, ~0ull, frame.acquire_semaphore, 0, &image_index);
    if (vkr == VK_ERROR_OUT_OF_DATE_KHR) {
        w->force_swapchain_recreate = true;
        return 0;
    }

    frame.current_image_index = image_index;
    frame.current_image = w->images[image_index];
    frame.current_image_view = w->image_views[image_index];

    return &frame;
}

VkResult Submit(const Frame& frame, const Context& vk, VkSubmitFlags submit_stage_mask) {
    VkSubmitInfo submit_info = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &frame.command_buffer;
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = &frame.acquire_semaphore;
    submit_info.pWaitDstStageMask = &submit_stage_mask;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &frame.release_semaphore;

    // @TODO: for multiwindow we either want to synchronize this or
    // allocate a queue per window (that should be ok for a few windows)
    return vkQueueSubmit(vk.queue, 1, &submit_info, frame.fence);
}

PresentResult PresentFrame(Window* w, Frame* frame, const Context& vk) {
    SwapchainStatus swapchain_status = SwapchainStatus::READY;

    VkPresentInfoKHR present_info = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &w->swapchain;
    present_info.pImageIndices = &frame->current_image_index;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = &frame->release_semaphore;
    VkResult vkr = vkQueuePresentKHR(vk.queue, &present_info);

    if (vkr == VK_ERROR_OUT_OF_DATE_KHR || vkr == VK_SUBOPTIMAL_KHR) {
        w->force_swapchain_recreate = true;
    } else if (vkr != VK_SUCCESS) {
        return PresentResult::FAILED;
    }

    w->swapchain_frame_index = (w->swapchain_frame_index + 1) % w->frames.length;
    frame->current_image = VK_NULL_HANDLE;
    frame->current_image_view = VK_NULL_HANDLE;
    frame->current_image_index = ~0ul;
    return PresentResult::SUCCESS;
}

Result
CreateWindowWithSwapchain(Window* w, const Context& vk, const char* name, u32 width, u32 height, bool verbose)
{
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
        return Result::SURFACE_CREATION_FAILED;
    }

    // Retrieve surface capabilities.
    VkSurfaceCapabilitiesKHR surface_capabilities = {};
    result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.physical_device, surface, &surface_capabilities);
    if (result != VK_SUCCESS) {
        return Result::SWAPCHAIN_CREATION_FAILED;
    }

    // Compute number of frames in flight.
    u32 num_frames = Max<u32>(2, surface_capabilities.minImageCount);
    if (surface_capabilities.maxImageCount > 0) {
        num_frames = Min<u32>(num_frames, surface_capabilities.maxImageCount);
    }

    // Retrieve supported surface formats.
    // @TODO: smarter format picking logic (HDR / non sRGB displays).
    u32 formats_count;
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(vk.physical_device, surface, &formats_count, 0);
    if (result != VK_SUCCESS || formats_count == 0) {
        return Result::SWAPCHAIN_CREATION_FAILED;
    }

    Array<VkSurfaceFormatKHR> formats(formats_count);
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(vk.physical_device, surface, &formats_count, formats.data);
    if (result != VK_SUCCESS || formats_count == 0) {
        return Result::SWAPCHAIN_CREATION_FAILED;
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

    Result res = CreateSwapchain(w, vk, surface, format, fb_width, fb_height, num_frames, VK_NULL_HANDLE);
    if (res != Result::SUCCESS) {
        return res;
    }

    // Create frames
    Array<Frame> frames(num_frames);
    for (u32 i = 0; i < num_frames; i++) {
        gfx::Frame& frame = frames[i];

        VkCommandPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;// | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = vk.queue_family_index;

        result = vkCreateCommandPool(vk.device, &pool_info, 0, &frame.command_pool);
        if (result != VK_SUCCESS) {
            return Result::API_OUT_OF_MEMORY;
        }

        VkCommandBufferAllocateInfo allocate_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        allocate_info.commandPool = frame.command_pool;
        allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate_info.commandBufferCount = 1;

        result = vkAllocateCommandBuffers(vk.device, &allocate_info, &frame.command_buffer);
        if (result != VK_SUCCESS) {
            return Result::API_OUT_OF_MEMORY;
        }

        VkFenceCreateInfo fence_info = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(vk.device, &fence_info, 0, &frame.fence);

        CreateGPUSemaphore(vk.device, &frame.acquire_semaphore);
        CreateGPUSemaphore(vk.device, &frame.release_semaphore);
    }

    w->window = window;
    w->surface = surface;
    w->swapchain_format = format;
    w->frames = std::move(frames);

    return Result::SUCCESS;
}

void
DestroyWindowWithSwapchain(Window* w, const Context& vk)
{
    // Swapchain
    for (usize i = 0; i < w->image_views.length; i++) {
        vkDestroyImageView(vk.device, w->image_views[i], 0);
    }
    vkDestroySwapchainKHR(vk.device, w->swapchain, 0);
    vkDestroySurfaceKHR(vk.instance, w->surface, 0);

    // Frames
    for (usize i = 0; i < w->image_views.length; i++) {
        gfx::Frame& frame = w->frames[i];
        vkDestroyFence(vk.device, frame.fence, 0);

        vkDestroySemaphore(vk.device, frame.acquire_semaphore, 0);
        vkDestroySemaphore(vk.device, frame.release_semaphore, 0);

        vkFreeCommandBuffers(vk.device, frame.command_pool, 1, &frame.command_buffer);
        vkDestroyCommandPool(vk.device, frame.command_pool, 0);
    }

}

struct Buffer
{
    VkBuffer buffer;
    VmaAllocation allocation;
};

struct BufferDesc {
    // Vulkan flags
    VkBufferUsageFlags usage;

    // VMA flags
    VmaAllocationCreateFlags alloc_flags = 0;
    VkMemoryPropertyFlags alloc_required_flags = 0;
    VkMemoryPropertyFlags alloc_preferred_flags = 0;
};

VkResult
CreateBuffer(Buffer* buffer, const Context& vk, size_t size, const BufferDesc&& desc) {
    // Alloc buffer
    VkBufferCreateInfo buffer_info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    buffer_info.size = size;
    buffer_info.usage = desc.usage;

    VmaAllocationCreateInfo alloc_create_info = {};
    alloc_create_info.requiredFlags = desc.alloc_required_flags;
    alloc_create_info.preferredFlags = desc.alloc_required_flags;
    alloc_create_info.flags = desc.alloc_flags;

    VkBuffer buf = 0;
    VmaAllocation allocation = {};
    VmaAllocationInfo alloc_info = {};
    VkResult vkr = vmaCreateBuffer(vk.vma, &buffer_info, &alloc_create_info, &buf, &allocation, &alloc_info);
    if (vkr != VK_SUCCESS) {
        return vkr;
    }

    buffer->buffer = buf;
    buffer->allocation = allocation;
    return VK_SUCCESS;
}

VkResult
CreateBufferFromData(Buffer* buffer, const Context& vk, ArrayView<u8> data, const BufferDesc&& desc) {
    assert(desc.alloc_required_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    VkResult vkr = CreateBuffer(buffer, vk, data.length, std::move(desc));
    if (vkr != VK_SUCCESS) {
        return vkr;
    }

    void* addr = 0;
    vkr = vmaMapMemory(vk.vma, buffer->allocation, &addr);
    if (vkr != VK_SUCCESS) {
        return vkr;
    }

    ArrayView<u8> map((u8*)addr, data.length);
    map.copy_exact(data);
    
    vmaUnmapMemory(vk.vma, buffer->allocation);

    return VK_SUCCESS;
}

void
DestroyBuffer(Buffer* buffer, const Context& vk)
{
    vmaDestroyBuffer(vk.vma, buffer->buffer, buffer->allocation);
}

struct Shader {
    VkShaderModule shader;
};

VkResult
CreateShader(Shader* shader, const Context& vk, ArrayView<u8> code) {
    VkShaderModuleCreateInfo module_info = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    module_info.codeSize = code.length;
    module_info.pCode = (u32*)code.data;
    VkShaderModule shader_module = 0;
    VkResult vkr = vkCreateShaderModule(vk.device, &module_info, 0, &shader_module);
    if (vkr != VK_SUCCESS) {
        return vkr;
    }

    shader->shader = shader_module;
    return VK_SUCCESS;
}

void
DestroyShader(Shader* shader, const Context& vk) {
    vkDestroyShaderModule(vk.device, shader->shader, NULL);
}


struct GraphicsPipeline {
    VkPipeline pipeline;
    VkPipelineLayout layout;
};

struct PipelineStageDesc {
    Shader shader;
    VkShaderStageFlagBits stage;
    const char* entry = "main";
};

// NOTE: struct layout of this must exactly match vulkan struct
struct VertexBindingDesc {
    u32 binding; // Potentially implicitly increasing?
    u32 stride;
    VkVertexInputRate input_rate = VK_VERTEX_INPUT_RATE_VERTEX;
};

// NOTE: struct layout of this must exactly match vulkan struct
struct VertexAttributeDesc {
    u32 location; // Potentially implicitly increasing?
    u32 binding;
    VkFormat format;
    u32 offset = 0;
};

struct InputAssemblyDesc {
    bool primitive_restart_enable = false;
    VkPrimitiveTopology primitive_topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
};

struct RasterizationDesc {
    VkPolygonMode polygon_mode = VK_POLYGON_MODE_FILL;
    VkCullModeFlags cull_mode = VK_CULL_MODE_NONE;
    VkFrontFace front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    bool depth_bias_enable = false;
    bool depth_clamp_enable = false;
    float line_width = 1.0f;
};

struct DepthDesc {
    bool test = false;
    bool write = false;
    VkCompareOp op = VK_COMPARE_OP_LESS;
    VkFormat format;
};

struct StencilDesc {
    bool test = false;
};

struct AttachmentDesc {
    VkFormat format;
    VkBool32 blend_enable = false;
    VkBlendFactor src_color_blend_factor = VK_BLEND_FACTOR_ZERO;
    VkBlendFactor dst_color_blend_factor = VK_BLEND_FACTOR_ZERO;
    VkBlendOp color_blend_op = VK_BLEND_OP_ADD;
    VkBlendFactor src_alpha_blend_factor = VK_BLEND_FACTOR_ZERO;
    VkBlendFactor dst_alpha_blend_factor = VK_BLEND_FACTOR_ZERO;
    VkBlendOp alpha_blend_op = VK_BLEND_OP_ADD;
    VkColorComponentFlags color_write_mask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
};

// NOTE: struct layout of this must exactly match vulkan struct
struct PushConstantRangeDesc {
    VkShaderStageFlagBits flags = VK_SHADER_STAGE_ALL;
    u32 offset;
    u32 size;
};

struct GraphicsPipelineDesc {
    Span<PipelineStageDesc> stages;
    Span<VertexBindingDesc> vertex_bindings;
    Span<VertexAttributeDesc> vertex_attributes;
    InputAssemblyDesc input_assembly;
    RasterizationDesc rasterization;
    DepthDesc depth;
    StencilDesc stencil;
    Span<PushConstantRangeDesc> push_constants;
    Span<VkDescriptorSetLayout> descriptor_sets;
    Span<AttachmentDesc> attachments;
};

VkResult
CreateGraphicsPipeline(GraphicsPipeline* graphics_pipeline, const Context& vk, const GraphicsPipelineDesc&& desc)
{
    Array<VkPipelineShaderStageCreateInfo> stages(desc.stages.length);
    for (usize i = 0; i < stages.length; i++) {
        stages[i] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        stages[i].flags = 0;
        stages[i].stage = desc.stages[i].stage;
        stages[i].pName = desc.stages[i].entry;
        stages[i].module = desc.stages[i].shader.shader;
    }

    VkPipelineVertexInputStateCreateInfo vertex_input_state = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    vertex_input_state.vertexBindingDescriptionCount = (u32)desc.vertex_bindings.length;
    vertex_input_state.pVertexBindingDescriptions = (VkVertexInputBindingDescription*)desc.vertex_bindings.data;
    vertex_input_state.vertexAttributeDescriptionCount = (u32)desc.vertex_attributes.length;
    vertex_input_state.pVertexAttributeDescriptions = (VkVertexInputAttributeDescription*)desc.vertex_attributes.data;

    VkPipelineInputAssemblyStateCreateInfo input_assembly_state = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    input_assembly_state.primitiveRestartEnable = desc.input_assembly.primitive_restart_enable;
    input_assembly_state.topology = desc.input_assembly.primitive_topology;

    VkPipelineTessellationStateCreateInfo tessellation_state = { VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO };

    VkPipelineViewportStateCreateInfo viewport_state = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterization_state = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rasterization_state.depthClampEnable = desc.rasterization.depth_clamp_enable;
    rasterization_state.rasterizerDiscardEnable = false;
    rasterization_state.polygonMode = desc.rasterization.polygon_mode;
    rasterization_state.cullMode = desc.rasterization.cull_mode;
    rasterization_state.frontFace = desc.rasterization.front_face;
    rasterization_state.depthBiasEnable = desc.rasterization.depth_bias_enable;
    rasterization_state.lineWidth = desc.rasterization.line_width;

    VkPipelineMultisampleStateCreateInfo multisample_state = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    multisample_state.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth_stencil_state = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    depth_stencil_state.depthTestEnable = desc.depth.test;
    depth_stencil_state.depthWriteEnable = desc.depth.write;

    Array<VkPipelineColorBlendAttachmentState> attachments(desc.attachments.length);
    for (usize i = 0; i < attachments.length; i++)
    {
        attachments[i].blendEnable         = desc.attachments[i].blend_enable;
        attachments[i].srcColorBlendFactor = desc.attachments[i].src_color_blend_factor;
        attachments[i].dstColorBlendFactor = desc.attachments[i].dst_color_blend_factor;
        attachments[i].colorBlendOp        = desc.attachments[i].color_blend_op;
        attachments[i].srcAlphaBlendFactor = desc.attachments[i].src_alpha_blend_factor;
        attachments[i].dstAlphaBlendFactor = desc.attachments[i].dst_alpha_blend_factor;
        attachments[i].alphaBlendOp        = desc.attachments[i].alpha_blend_op;
        attachments[i].colorWriteMask      = desc.attachments[i].color_write_mask;
    }

    Array<VkFormat> attachment_formats(desc.attachments.length);
    for (usize i = 0; i < attachments.length; i++)
    {
        attachment_formats[i] = desc.attachments[i].format;
    }

    VkPipelineColorBlendStateCreateInfo blend_state = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    blend_state.attachmentCount = (u32)attachments.length;
    blend_state.pAttachments = attachments.data;

    VkDynamicState dynamic_states[2] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    VkPipelineDynamicStateCreateInfo dynamic_state = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynamic_state.dynamicStateCount = ArrayCount(dynamic_states);
    dynamic_state.pDynamicStates = dynamic_states;

    // Rendering
    VkPipelineRenderingCreateInfo rendering_create_info = { VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    rendering_create_info.colorAttachmentCount = (u32)attachment_formats.length;
    rendering_create_info.pColorAttachmentFormats = attachment_formats.data;
    rendering_create_info.depthAttachmentFormat = desc.depth.format;

    // Pipeline layout
    VkPipelineLayoutCreateInfo layout_info = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    layout_info.setLayoutCount = (u32)desc.descriptor_sets.length;
    layout_info.pSetLayouts = desc.descriptor_sets.data;
    layout_info.pushConstantRangeCount = (u32)desc.push_constants.length;
    layout_info.pPushConstantRanges = (VkPushConstantRange*)desc.push_constants.data;

    VkResult vkr;

    VkPipelineLayout layout = 0;
    vkr = vkCreatePipelineLayout(vk.device, &layout_info, 0, &layout);
    if (vkr != VK_SUCCESS) {
        return vkr;
    }

    // Graphics Pipeline
    VkGraphicsPipelineCreateInfo pipeline_create_info = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pipeline_create_info.pNext = &rendering_create_info;
    pipeline_create_info.flags = 0;

    // Shaders
    pipeline_create_info.stageCount = (u32)stages.length;
    pipeline_create_info.pStages = stages.data;

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
    if (vkr != VK_SUCCESS) {
        return vkr;
    }
    
    graphics_pipeline->pipeline = pipeline;
    graphics_pipeline->layout = layout;

    return VK_SUCCESS;
}

void
DestroyGraphicsPipeline(GraphicsPipeline* pipeline, const Context& vk) {
    vkDestroyPipelineLayout(vk.device, pipeline->layout, 0);
    vkDestroyPipeline(vk.device, pipeline->pipeline, 0);
    *pipeline = {};
}

VkResult
CreateDepthBuffer(DepthBuffer* depth_buffer, const Context& vk, u32 width, u32 height)
{
    // Create a depth buffer.
    VkImageCreateInfo image_create_info = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    image_create_info.imageType = VK_IMAGE_TYPE_2D;
    image_create_info.extent.width = width;
    image_create_info.extent.height = height;
    image_create_info.extent.depth = 1;
    image_create_info.mipLevels = 1;
    image_create_info.arrayLayers = 1;
    image_create_info.format = VK_FORMAT_D32_SFLOAT;
    image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_create_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;

    VmaAllocationCreateInfo alloc_create_info = {};
    alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_create_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    alloc_create_info.priority = 1.0f;

    VkImage image;
    VmaAllocation allocation;
    VkResult vkr = vmaCreateImage(vk.vma, &image_create_info, &alloc_create_info, &image, &allocation, nullptr);
    if (vkr != VK_SUCCESS) {
        return vkr;
    }

    VkImageViewCreateInfo image_view_info = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    image_view_info.image = image;
    image_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    image_view_info.format = VK_FORMAT_D32_SFLOAT;
    image_view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    image_view_info.subresourceRange.levelCount = 1;
    image_view_info.subresourceRange.layerCount = 1;

    VkImageView image_view = 0;
    vkr = vkCreateImageView(vk.device, &image_view_info, 0, &image_view);
    if (vkr != VK_SUCCESS) {
        return vkr;
    }

    depth_buffer->image = image;
    depth_buffer->view = image_view;
    depth_buffer->allocation = allocation;

    return vkr;
}

void
DestroyDepthBuffer(const Context& vk, DepthBuffer* depth_buffer)
{
    vkDestroyImageView(vk.device, depth_buffer->view, 0);
    vmaDestroyImage(vk.vma, depth_buffer->image, depth_buffer->allocation);
    *depth_buffer = {};
}

struct BindlessDescriptorSet
{
    VkDescriptorSet set;
    VkDescriptorSetLayout layout;
    VkDescriptorPool pool;
};

struct BindlessDescriptorSetEntryDesc
{
    u32 count;
    VkDescriptorType type;
};

struct BindlessDescriptorSetDesc
{
    Span<BindlessDescriptorSetEntryDesc> entries;
};

void CreateBindlessDescriptorSet(BindlessDescriptorSet* set, const Context& vk, const BindlessDescriptorSetDesc&& desc)
{
    usize N = desc.entries.length;
    Array<VkDescriptorSetLayoutBinding> bindings(N);
    Array<VkDescriptorBindingFlags> flags(N);
    Array<VkDescriptorPoolSize> descriptor_pool_sizes(N);

    for (uint32_t i = 0; i < N; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = desc.entries[i].type;
        bindings[i].descriptorCount = desc.entries[i].count;
        bindings[i].stageFlags = VK_SHADER_STAGE_ALL;
        flags[i] =
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
        descriptor_pool_sizes[i] = { desc.entries[i].type, desc.entries[i].count };
    }

    VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
    binding_flags.bindingCount = (uint32_t)bindings.length;
    binding_flags.pBindingFlags = flags.data;

    VkDescriptorSetLayoutCreateInfo create_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    create_info.pNext = &binding_flags;
    create_info.bindingCount = (uint32_t)bindings.length;
    create_info.pBindings = bindings.data;
    create_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;

    // Create layout
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    vkCreateDescriptorSetLayout(vk.device, &create_info, 0, &descriptor_layout);

    // Create pool
    VkDescriptorPoolCreateInfo descriptor_pool_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    descriptor_pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    descriptor_pool_info.maxSets = 1;
    descriptor_pool_info.pPoolSizes = descriptor_pool_sizes.data;
    descriptor_pool_info.poolSizeCount = (uint32_t)descriptor_pool_sizes.length;

    VkDescriptorPool descriptor_pool = 0;
    vkCreateDescriptorPool(vk.device, &descriptor_pool_info, 0, &descriptor_pool);

    // Create descriptor set
    VkDescriptorSetAllocateInfo allocate_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocate_info.descriptorPool = descriptor_pool;
    allocate_info.pSetLayouts = &descriptor_layout;
    allocate_info.descriptorSetCount = 1;

    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    vkAllocateDescriptorSets(vk.device, &allocate_info, &descriptor_set);

    set->set = descriptor_set;
    set->layout = descriptor_layout;
    set->pool = descriptor_pool;
}

void
DestroyBindlessDescriptorSet(BindlessDescriptorSet* bindless, const Context& vk)
{
    vkDestroyDescriptorPool(vk.device, bindless->pool, 0);
    vkDestroyDescriptorSetLayout(vk.device, bindless->layout, 0);
    *bindless = {};
}

}
