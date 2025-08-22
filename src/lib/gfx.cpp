#define VOLK_IMPLEMENTATION
#define VMA_IMPLEMENTATION

#include <xpg/gfx.h>
#include <xpg/log.h>
#include <xpg/platform.h>

#include <vulkan/vk_enum_string_helper.h>   // Vulkan helper strings for printing

#define XPG_VERSION 0

// #define COPY_QUEUE_INDEX 2

// FEATURE: this technically works, but applications need to be aware of this.
// When using this, application resize logic is much more complicated, as it needs
// to delay freeing objects in use. This appears to also fix flickering / artifacts
// on image resize.
//
// Apps also need to worry about recreation of new objects with the new resolution.
// There are two cases here:
// - single buffered objects (e.g. color and depth attachments)
//      -> can recreate immediately but need to enqueue old for freeing
// - per frame buffered objects (e.g. upload buffers, swapchain framebuffers when using render passes) that depend on swapchain resolution
//      -> can either recreate all at once and treat as above, or one per frame to limit peak memory usage
//
// App needs to know the index of the dead frames after resizing, and potentially helpers for making both cases
// easy to handle. Helpers could potentially be callback based? Probably control flow is easier if wait returns some
// info that can be used to free this stuff.
//
// In theory we could have this be a creation parameter and have apps make the choice (maybe even at runtime), but it could be hard to
// support both recreation logics in app, can maybe make this not that bad if helpers are good.
#define SYNC_SWAPCHAIN_DESTRUCTION 1

namespace xpg {
namespace gfx {

#define DEBUG_UTILS_OBJECT_NAME(typ, obj, name) \
    if (obj) { \
        name_info.objectType = typ; \
        name_info.objectHandle = (u64)obj; \
        name_info.pObjectName = name; \
        vkSetDebugUtilsObjectNameEXT(device, &name_info); \
    }


static VkBool32 VKAPI_CALL
VulkanDebugUtilsMessengerCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT           message_severity,
    VkDebugUtilsMessageTypeFlagsEXT                  message_types,
    const VkDebugUtilsMessengerCallbackDataEXT*      callback_data,
    void*                                            user_data)
{
    if      (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)   logging::error  ("gfx/validation", "%s", callback_data->pMessage);
    else if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) logging::warning("gfx/validation", "%s", callback_data->pMessage);
    else if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)    logging::info   ("gfx/validation", "%s", callback_data->pMessage);
    else if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) logging::debug  ("gfx/validation", "%s", callback_data->pMessage);

#ifdef _WIN32
    const char* type =
        (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        ? "ERROR"
        : (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        ? "WARNING"
        : (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
        ? "INFO"
        : "VERBOSE";
    char message[4096];
    snprintf(message, sizeof(message), "%s: %s\n", type, callback_data->pMessage);
    OutputDebugStringA(message);
#endif

    // Uncomment to assert on validation errors
    // if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) assert(!"Validation error encountered!");

    return VK_FALSE;
}

// Error callback function
static void Callback_Error(int error, const char* description) {
    // Print the error code and description to stdout
    logging::error("glfw", "[%d]: %s", error, description);
}

Result
Init() {
    glfwSetErrorCallback(Callback_Error);
#ifdef XPG_MOLTENVK_STATIC
    glfwInitVulkanLoader(vkGetInstanceProcAddr);
#endif
    glfwInit();
    return Result::SUCCESS;
}

void Callback_WindowRefresh(GLFWwindow* window) {
    Window* w = (Window*)glfwGetWindowUserPointer(window);
    if (w && w->callbacks.draw) {
        w->callbacks.draw();
    }

    if(w && w->callbacks.prev_callback_window_refresh) {
        w->callbacks.prev_callback_window_refresh(window);
    }
}

void Callback_MouseButton(GLFWwindow* window, int button, int action, int mods) {
    Window* w = (Window*)glfwGetWindowUserPointer(window);
    if (w && w->callbacks.mouse_button_event) {
        glm::dvec2 pos = {};
        glfwGetCursorPos(window, &pos.x, &pos.y);
        w->callbacks.mouse_button_event((glm::ivec2)pos, (MouseButton)button, (Action)action, (Modifiers)mods);
    }

    if(w && w->callbacks.prev_callback_mouse_button) {
        w->callbacks.prev_callback_mouse_button(window, button, action, mods);
    }
}

void Callback_Scroll(GLFWwindow* window, double x, double y) {
    Window* w = (Window*)glfwGetWindowUserPointer(window);
    if (w && w->callbacks.mouse_scroll_event) {
        glm::dvec2 pos = {};
        glfwGetCursorPos(window, &pos.x, &pos.y);
        w->callbacks.mouse_scroll_event((glm::ivec2)pos, glm::ivec2((s32)x, (s32)y));
    }

    if(w && w->callbacks.prev_callback_scroll) {
        w->callbacks.prev_callback_scroll(window, x, y);
    }
}

void Callback_CursorPos(GLFWwindow* window, double x, double y) {
    Window* w = (Window*)glfwGetWindowUserPointer(window);
    if (w && w->callbacks.mouse_move_event) {
        w->callbacks.mouse_move_event(glm::ivec2((s32)x, (s32)y));
    }

    if(w && w->callbacks.prev_callback_cursor_pos) {
        w->callbacks.prev_callback_cursor_pos(window, x, y);
    }
}

void Callback_Key(GLFWwindow* window, int key, int scancode, int action, int mods) {
    Window* w = (Window*)glfwGetWindowUserPointer(window);
    if (w && w->callbacks.key_event) {
        w->callbacks.key_event((Key)key, (Action)action, (Modifiers)mods);
    }

    if(w && w->callbacks.prev_callback_key) {
        w->callbacks.prev_callback_key(window, key, scancode, action, mods);
    }
}

#ifdef _WIN32
DWORD WINAPI thread_proc(void* param) {
    HWND window = (HWND)param;
    while (true) {
        SendMessage(window, WM_PAINT, 0, 0);
    }
    return 0;
}
#endif

void SetWindowCallbacks(Window* window, WindowCallbacks&& callbacks) {
// #ifdef _WIN32
//     // Redraw during move / resize
//     HWND hwnd = glfwGetWin32Window(window->window);
//     HANDLE thread = CreateThread(0, 0, thread_proc, hwnd, 0, 0);
//     if (thread) {
//         CloseHandle(thread);
//     }
// #endif

    window->callbacks = move(callbacks);
    glfwSetWindowUserPointer(window->window, window);

    // If installing multiple times, do not install recursively
#define SAVE_CALLBACK_WITHOUT_RECURSION(type, name, glfw_name, callback) \
    type name = glfw_name(window->window, callback); \
    if (name != callback) { \
        window->callbacks.name = name; \
    }

    SAVE_CALLBACK_WITHOUT_RECURSION(GLFWwindowrefreshfun, prev_callback_window_refresh, glfwSetWindowRefreshCallback, Callback_WindowRefresh);
    SAVE_CALLBACK_WITHOUT_RECURSION(GLFWmousebuttonfun, prev_callback_mouse_button, glfwSetMouseButtonCallback, Callback_MouseButton);
    SAVE_CALLBACK_WITHOUT_RECURSION(GLFWscrollfun, prev_callback_scroll, glfwSetScrollCallback, Callback_Scroll);
    SAVE_CALLBACK_WITHOUT_RECURSION(GLFWcursorposfun, prev_callback_cursor_pos, glfwSetCursorPosCallback, Callback_CursorPos);
    SAVE_CALLBACK_WITHOUT_RECURSION(GLFWkeyfun, prev_callback_key, glfwSetKeyCallback, Callback_Key);
}

void ProcessEvents(bool block) {
    if (block) {
        glfwWaitEvents();
    }
    else {
        glfwPollEvents();
    }
}

bool ShouldClose(const Window& window) {
    return glfwWindowShouldClose(window.window);
}

Modifiers GetModifiersState(const Window& window) {
    u32 m = 0;
    m |= glfwGetKey(window.window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS || glfwGetKey(window.window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS ? (u32)Modifiers::Ctrl  : (u32)Modifiers::None;
    m |= glfwGetKey(window.window, GLFW_KEY_LEFT_SHIFT)   == GLFW_PRESS || glfwGetKey(window.window, GLFW_KEY_RIGHT_SHIFT)   == GLFW_PRESS ? (u32)Modifiers::Shift : (u32)Modifiers::None;
    m |= glfwGetKey(window.window, GLFW_KEY_LEFT_ALT)     == GLFW_PRESS || glfwGetKey(window.window, GLFW_KEY_RIGHT_ALT)     == GLFW_PRESS ? (u32)Modifiers::Alt   : (u32)Modifiers::None;
    m |= glfwGetKey(window.window, GLFW_KEY_LEFT_SUPER)   == GLFW_PRESS || glfwGetKey(window.window, GLFW_KEY_RIGHT_SUPER)   == GLFW_PRESS ? (u32)Modifiers::Super : (u32)Modifiers::None;
    return (Modifiers)m;
}

void CloseWindow(const Window& window) {
    glfwSetWindowShouldClose(window.window, 1);
}

struct GenericFeatureStruct {
    VkStructureType sType;
    void* pNext;
    VkBool32 values[1];
};

template<size_t F, size_t E>
struct FeatureAndExtensionDependencies {
    DeviceFeatures::Flags flag;
    GenericFeatureStruct* features_req[F];
    GenericFeatureStruct* features_sup[F];
    const char*  extensions[E];
};

template<size_t E>
struct ExtensionDependencies {
    DeviceFeatures::Flags flag;
    const char*  extensions[E];
};

struct PhysicalDeviceInfo {
    u32 device_api_version = 0;
    VkPhysicalDeviceType device_type = VK_PHYSICAL_DEVICE_TYPE_OTHER;

    DeviceFeatures supported_features = DeviceFeatures::NONE;

    u32 queue_family_index = VK_QUEUE_FAMILY_IGNORED;
    u32 compute_queue_family_index = VK_QUEUE_FAMILY_IGNORED;
    u32 copy_queue_family_index = VK_QUEUE_FAMILY_IGNORED;

    float timestamp_period = false;
    bool queue_timestamp_queries = false;
    bool compute_queue_timestamp_queries = false;
    bool copy_queue_timestamp_queries = false;
    bool require_portability_subset_extension = false;
};

template <typename T>
void Chain(void** parent_next, T* child) {
    static_assert(sizeof(T) >= sizeof(GenericFeatureStruct), "Feature struct must be equal or larger than GenericFeatureStruct");

    child->pNext = *parent_next;
    *parent_next = child;
}

template <typename T>
void ClearFeatures(T& features) {
    static_assert(sizeof(T) >= sizeof(GenericFeatureStruct), "Feature struct must be equal or larger than GenericFeatureStruct");
    static_assert((sizeof(T) - offsetof(GenericFeatureStruct, values)) % sizeof(VkBool32) == 0, "Fields in feature struct must have size that is a multiple of boolean size");

    GenericFeatureStruct* feat = (GenericFeatureStruct*)&features;
    constexpr size_t count = (sizeof(T) - offsetof(GenericFeatureStruct, values)) / sizeof(VkBool32);
    for(usize i = 0; i < count ; i++) {
        feat->values[i] = VK_FALSE;
    }
}

template <typename T>
bool CheckAllSupported(const T& requested, const T& supported) {
    static_assert(sizeof(T) >= sizeof(GenericFeatureStruct), "Feature struct must be equal or larger than GenericFeatureStruct");
    static_assert((sizeof(T) - offsetof(GenericFeatureStruct, values)) % sizeof(VkBool32) == 0, "Fields in feature struct must have size that is a multiple of boolean size");

    assert(requested.sType == supported.sType);

    GenericFeatureStruct* req = (GenericFeatureStruct*)&requested;
    GenericFeatureStruct* sup = (GenericFeatureStruct*)&supported;

    constexpr size_t count = (sizeof(T) - offsetof(GenericFeatureStruct, values)) / sizeof(VkBool32);

    bool all_supported = true;
    for(usize i = 0; i < count ; i++) {
        all_supported = all_supported && (!req->values[i] || sup->values[i]);
    }
    return all_supported;
}


Result
CreateContext(Context* vk, const ContextDesc&& desc)
{
    VkResult result;

#ifndef XPG_MOLTENVK_STATIC
    // Initialize vulkan loader.
    result = volkInitialize();
    if (result != VK_SUCCESS) {
        logging::error("gfx", "volkInitialize failed: %d", result);
        return Result::API_ERROR;
    }
#endif

    // Query vulkan version. If this is not available it's 1.0
    u32 instance_version = VK_API_VERSION_1_0;
    if(vkEnumerateInstanceVersion) {
        result = vkEnumerateInstanceVersion(&instance_version);
        if (result != VK_SUCCESS) {
            logging::error("gfx/instance", "vkEnumerateInstanceVersion failed: %d", result);
            return Result::API_OUT_OF_MEMORY;
        }
    }

    u32 instance_version_major = VK_API_VERSION_MAJOR(instance_version);
    u32 instance_version_minor = VK_API_VERSION_MINOR(instance_version);
    u32 instance_version_patch = VK_API_VERSION_PATCH(instance_version);
    logging::info("gfx/instance", "Vulkan instance API version %u.%u.%u", instance_version_major, instance_version_minor, instance_version_patch);

    // Enumerate instance extensions.
    uint32_t instance_extension_count;
    result = vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count, nullptr);
    if (result != VK_SUCCESS) {
        logging::error("gfx/instance", "vkEnumerateInstanceLayerProperties for count failed: %d", result);
        return Result::API_OUT_OF_MEMORY;
    }

    Array<VkExtensionProperties> instance_extensions(instance_extension_count);
    result = vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count, instance_extensions.data);

    // Enumerate layer properties.
    u32 layer_properties_count = 0;
    result = vkEnumerateInstanceLayerProperties(&layer_properties_count, NULL);
    if (result != VK_SUCCESS) {
        logging::error("gfx/instance", "vkEnumerateInstanceLayerProperties for count failed: %d", result);
        return Result::API_OUT_OF_MEMORY;
    }

    Array<VkLayerProperties> layer_properties(layer_properties_count);
    result = vkEnumerateInstanceLayerProperties(&layer_properties_count, layer_properties.data);
    if (result != VK_SUCCESS) {
        logging::error("gfx/instance", "vkEnumerateInstanceLayerProperties for values failed: %d", result);
        return Result::API_OUT_OF_MEMORY;
    }

    // Application info.
    VkApplicationInfo application_info = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    application_info.apiVersion = desc.minimum_api_version;
    application_info.pApplicationName = "XPG";
    application_info.applicationVersion = XPG_VERSION;

    // Instance extensions
    Array<const char*> enabled_instance_extensions;
    if (desc.require_presentation) {
        u32 glfw_instance_extensions_count;
        const char** glfw_instance_extensions = glfwGetRequiredInstanceExtensions(&glfw_instance_extensions_count);
        for (u32 i = 0; i < glfw_instance_extensions_count; i++) {
            enabled_instance_extensions.add(glfw_instance_extensions[i]);
        }
    }

    // Enable portability extension if present (required by MoltenVK)
    u32 instance_create_flags = 0;
    for(usize i = 0; i < instance_extensions.length; i++) {
        if (strcmp(instance_extensions[i].extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0) {
            logging::info("gfx/instance", "VK_KHR_portability_enumeration available");
            enabled_instance_extensions.add(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            instance_create_flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
            break;
        }
    }

    // Enable debug utils if requested and available. If validation is requested we also implicitly request this.
    bool debug_utils_requested = desc.enable_debug_utils || desc.enable_validation_layer;
    bool debug_utils_enabled = false;
    if (debug_utils_requested) {
        bool debug_utils_available = false;
        for (usize i = 0; i < instance_extensions.length; i++) {
            if (strcmp(instance_extensions[i].extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0)
            {
                debug_utils_available = true;
            }
        }

        if (debug_utils_available) {
            enabled_instance_extensions.add(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            debug_utils_enabled = true;
        } else {
            logging::warning("gfx/instance", "Debug utils extension requested, but not available");
        }
    }

    // Instance info.
    VkInstanceCreateInfo info = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    info.enabledExtensionCount = (u32)enabled_instance_extensions.length;
    info.ppEnabledExtensionNames = enabled_instance_extensions.data;
    info.pApplicationInfo = &application_info;
    info.flags = instance_create_flags;

    // Check if validation layer is present.
    const char* validation_layer_name = "VK_LAYER_KHRONOS_validation";
    bool validation_layer_enabled = false;
    if (desc.enable_validation_layer) {
        bool validation_layer_present = false;
        for (u32 i = 0; i < layer_properties_count; i++) {
            VkLayerProperties& l = layer_properties[i];
            if (strcmp(l.layerName, validation_layer_name) == 0) {
                validation_layer_present = true;
                logging::info("gfx/layers", "Vulkan validation layer found");
                break;
            }
        }

        if (validation_layer_present) {
            validation_layer_enabled = true;
        } else {
            logging::warning("gfx/layers", "Validation layer requested, but not available");
        }
    }

    const char* enabled_layers[1] = { validation_layer_name };

    if (validation_layer_enabled) {
        info.enabledLayerCount = ArrayCount(enabled_layers);
        info.ppEnabledLayerNames = enabled_layers;
    }

    const VkLayerSettingEXT layer_settings[] = {
        { "VK_LAYER_KHRONOS_validation", "validate_sync", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &desc.enable_synchronization_validation },
        { "VK_LAYER_KHRONOS_validation", "gpuav_enable", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &desc.enable_gpu_based_validation },
    };

    VkLayerSettingsCreateInfoEXT layer_settings_create_info = {VK_STRUCTURE_TYPE_LAYER_SETTINGS_CREATE_INFO_EXT};
    layer_settings_create_info.settingCount = ArrayCount(layer_settings);
    layer_settings_create_info.pSettings = layer_settings;
    if (validation_layer_enabled) {
        info.pNext = &layer_settings_create_info;
    }

    // Create instance.
    VkInstance instance = 0;
    result = vkCreateInstance(&info, 0, &instance);
    if (result != VK_SUCCESS) {
        logging::error("gfx/instance", "vkCreateInstance failed: %d", result);

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
#ifndef XPG_MOLTENVK_STATIC
    volkLoadInstance(instance);
#endif

    // Install debug callback.
    VkDebugUtilsMessengerEXT debug_messenger = 0;
    if (debug_utils_enabled) {
        VkDebugUtilsMessengerCreateInfoEXT messenger_create_info = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
        messenger_create_info.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        messenger_create_info.messageType =
            // VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        messenger_create_info.pfnUserCallback = VulkanDebugUtilsMessengerCallback;

        result = vkCreateDebugUtilsMessengerEXT(instance, &messenger_create_info, 0, &debug_messenger);
        if (result != VK_SUCCESS) {
            // Failed to install debug callback.
            logging::warning("gfx/debug", "vkCreateDebugUtilsMessengerEXT failed: %d", result);
        }
    }

    // Requested device features
    DeviceFeatures features_to_check = desc.required_features | desc.optional_features;
    if (desc.required_features & (DeviceFeatures::RAY_TRACING_PIPELINE | DeviceFeatures::RAY_QUERY)) {
        features_to_check = features_to_check | DeviceFeatures::DESCRIPTOR_INDEXING;
    }

    void* sup_next = {};

    #define CHAIN(o, flags) \
        decltype(o) o##_sup = { o.sType }; \
        bool o##_sup_check = false; \
        if (features_to_check & ((DeviceFeatures)flags)) { \
            o##_sup_check = true; \
            Chain(&sup_next, &o##_sup); \
        }

    // Dynamic rendering
    VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamic_rendering_features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR};
    dynamic_rendering_features.dynamicRendering = VK_TRUE;
    CHAIN(dynamic_rendering_features, DeviceFeatures::DYNAMIC_RENDERING);

    // Synchronization 2
    VkPhysicalDeviceSynchronization2FeaturesKHR synchronization_2_features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR};
    synchronization_2_features.synchronization2 = true;
    CHAIN(synchronization_2_features, DeviceFeatures::SYNCHRONIZATION_2);

    // Scalar block layout
    VkPhysicalDeviceScalarBlockLayoutFeaturesEXT scalar_block_layout_features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES_EXT};
    scalar_block_layout_features.scalarBlockLayout = true;
    CHAIN(scalar_block_layout_features, DeviceFeatures::SCALAR_BLOCK_LAYOUT);

    // Descriptor indexing
    VkPhysicalDeviceDescriptorIndexingFeaturesEXT descriptor_indexing_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT};
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
    CHAIN(descriptor_indexing_features, DeviceFeatures::DESCRIPTOR_INDEXING);

    // Raytracing
    VkPhysicalDeviceBufferDeviceAddressFeaturesKHR buffer_device_address_features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR};
    buffer_device_address_features.bufferDeviceAddress = VK_TRUE;
    CHAIN(buffer_device_address_features, DeviceFeatures::RAY_TRACING_PIPELINE | DeviceFeatures::RAY_QUERY);

    VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure_features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    acceleration_structure_features.accelerationStructure = VK_TRUE;
    CHAIN(acceleration_structure_features, DeviceFeatures::RAY_TRACING_PIPELINE | DeviceFeatures::RAY_QUERY);

    // - Ray query
    VkPhysicalDeviceRayQueryFeaturesKHR ray_query_features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    ray_query_features.rayQuery = VK_TRUE;
    CHAIN(ray_query_features, DeviceFeatures::RAY_QUERY);

    // - Ray tracing pipeline
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR ray_tracing_pipeline_features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
    ray_tracing_pipeline_features.rayTracingPipeline = VK_TRUE;
    CHAIN(ray_tracing_pipeline_features, DeviceFeatures::RAY_TRACING_PIPELINE);

    // Host query reset
    VkPhysicalDeviceHostQueryResetFeaturesEXT host_query_reset_features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES_EXT };
    host_query_reset_features.hostQueryReset = VK_TRUE;
    CHAIN(host_query_reset_features, DeviceFeatures::HOST_QUERY_RESET);

    // Timeline semaphores
    VkPhysicalDeviceTimelineSemaphoreFeaturesKHR timeline_semaphore_features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR };
    timeline_semaphore_features.timelineSemaphore = VK_TRUE;
    CHAIN(timeline_semaphore_features, DeviceFeatures::TIMELINE_SEMAPHORES);

    // Feature dependencies
    FeatureAndExtensionDependencies<1, 3> dynamic_rendering_deps = {
        .flag = DeviceFeatures::DYNAMIC_RENDERING,
        .features_req = { (GenericFeatureStruct*)&dynamic_rendering_features },
        .features_sup = { (GenericFeatureStruct*)&dynamic_rendering_features_sup },
        .extensions = {
            VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME,
            VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME,
            VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME
        },
    };

    FeatureAndExtensionDependencies<1, 1> synchronization_2_deps = {
        .flag = DeviceFeatures::SYNCHRONIZATION_2,
        .features_req = { (GenericFeatureStruct*)&synchronization_2_features },
        .features_sup = { (GenericFeatureStruct*)&synchronization_2_features_sup },
        .extensions = { VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME },
    };

    FeatureAndExtensionDependencies<1, 1> scalar_block_layout_deps = {
        .flag = DeviceFeatures::SCALAR_BLOCK_LAYOUT,
        .features_req = { (GenericFeatureStruct*)&scalar_block_layout_features },
        .features_sup = { (GenericFeatureStruct*)&scalar_block_layout_features_sup },
        .extensions = { VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME },
    };

    FeatureAndExtensionDependencies<1, 1> descriptor_indexing_deps = {
        .flag = DeviceFeatures::DESCRIPTOR_INDEXING,
        .features_req = { (GenericFeatureStruct*)&descriptor_indexing_features },
        .features_sup = { (GenericFeatureStruct*)&descriptor_indexing_features_sup },
        .extensions = { VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME },
    };

    FeatureAndExtensionDependencies<4, 7> ray_query_deps = {
        .flag = DeviceFeatures::RAY_QUERY,
        .features_req = {
            (GenericFeatureStruct*)&descriptor_indexing_features,
            (GenericFeatureStruct*)&buffer_device_address_features,
            (GenericFeatureStruct*)&acceleration_structure_features,
            (GenericFeatureStruct*)&ray_query_features,
        },
        .features_sup = {
            (GenericFeatureStruct*)&descriptor_indexing_features_sup,
            (GenericFeatureStruct*)&buffer_device_address_features_sup,
            (GenericFeatureStruct*)&acceleration_structure_features_sup,
            (GenericFeatureStruct*)&ray_query_features_sup,
        },
        .extensions = {
            VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            VK_KHR_SPIRV_1_4_EXTENSION_NAME,
            VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
            VK_KHR_RAY_QUERY_EXTENSION_NAME,
        },
    };

    FeatureAndExtensionDependencies<4, 7> ray_tracing_pipeline_deps = {
        .flag = DeviceFeatures::RAY_TRACING_PIPELINE,
        .features_req = {
            (GenericFeatureStruct*)&descriptor_indexing_features,
            (GenericFeatureStruct*)&buffer_device_address_features,
            (GenericFeatureStruct*)&acceleration_structure_features,
            (GenericFeatureStruct*)&ray_tracing_pipeline_features,
        },
        .features_sup = {
            (GenericFeatureStruct*)&descriptor_indexing_features_sup,
            (GenericFeatureStruct*)&buffer_device_address_features_sup,
            (GenericFeatureStruct*)&acceleration_structure_features_sup,
            (GenericFeatureStruct*)&ray_tracing_pipeline_features_sup,
        },
        .extensions = {
            VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            VK_KHR_SPIRV_1_4_EXTENSION_NAME,
            VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
            VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        },
    };

    FeatureAndExtensionDependencies<1, 1> host_query_reset_deps = {
        .flag = DeviceFeatures::HOST_QUERY_RESET,
        .features_req = { (GenericFeatureStruct*)&host_query_reset_features },
        .features_sup = { (GenericFeatureStruct*)&host_query_reset_features_sup },
        .extensions = { VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME },
    };

    FeatureAndExtensionDependencies<1, 1> timeline_semaphore_deps = {
        .flag = DeviceFeatures::TIMELINE_SEMAPHORES,
        .features_req = { (GenericFeatureStruct*)&timeline_semaphore_features },
        .features_sup = { (GenericFeatureStruct*)&timeline_semaphore_features_sup },
        .extensions = { VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME },
    };

    ExtensionDependencies<2> external_resources_deps = {
        .flag = DeviceFeatures::EXTERNAL_RESOURCES,
        .extensions = {
#ifdef _WIN32
            VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
#else
            VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
#endif
        },
    };

    ExtensionDependencies<1> calibrated_timestamps_deps = {
        .flag = DeviceFeatures::CALIBRATED_TIMESTAMPS,
        .extensions = {
            VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME,
        },
    };

    // Enumerate and choose a physical devices.
    u32 physical_device_count = 0;
    result = vkEnumeratePhysicalDevices(instance, &physical_device_count, 0);
    if (result != VK_SUCCESS) {
        logging::error("gfx/device", "vkEnumeratePhysicalDevices for count failed: %d", result);
        return Result::API_OUT_OF_MEMORY;
    }

    Array<VkPhysicalDevice> physical_devices(physical_device_count);
    result = vkEnumeratePhysicalDevices(instance, &physical_device_count, physical_devices.data);
    if (result != VK_SUCCESS) {
        logging::error("gfx/device", "vkEnumeratePhysicalDevices for values failed: %d", result);
        return Result::API_OUT_OF_MEMORY;
    }


    u32 picked_index = physical_device_count;
    PhysicalDeviceInfo picked_info = {};

    for (u32 i = 0; i < physical_device_count; i++) {
        PhysicalDeviceInfo info = {};

        // Check device properties
        VkPhysicalDeviceProperties properties = {};
        vkGetPhysicalDeviceProperties(physical_devices[i], &properties);
        info.device_api_version = properties.apiVersion;

        // GPU type
        info.device_type = properties.deviceType;

        // Timetsamp support
        info.timestamp_period = properties.limits.timestampPeriod;

        // Check queues
        u32 queue_family_property_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[i], &queue_family_property_count, 0);
        Array<VkQueueFamilyProperties> queue_family_properties(queue_family_property_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[i], &queue_family_property_count, queue_family_properties.data);

        bool supports_presentation = false;
        for (u32 j = 0; j < queue_family_property_count; j++) {
            VkQueueFamilyProperties& prop = queue_family_properties[j];
            if(prop.queueCount > 0) {
                logging::trace("gfx/device", "Queue %d | flags: 0x%x", j, prop.queueFlags);
                if(prop.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                    if(info.queue_family_index == VK_QUEUE_FAMILY_IGNORED) {
                        // NOTE: Doing this properly requires a surface, to support this we would need to delay queue choice after creating a window.
                        // Order of dependencies is then: instance -> window -> physical device + queue family -> device -> everthing else
                        // There is actually a platform specific version of this that does not require a window:
                        //     vkGetPhysicalDeviceWin32PresentationSupportKHR
                        // This is conveniently wrapped for us by:
                        //     glfwGetPhysicalDevicePresentationSupport
                        // The spec says there is no guarantee of the queue then supporting all surfaces, but legend says
                        // this will not happen on our supported targets. We can also add the surface check and to window
                        // creation for logging / debugging, and instead trust the queue to work.
                        // The actual API would be:
                        //     VkBool32 presentationSupport = false;
                        //     vkGetPhysicalDeviceSurfaceSupportKHR(physical_devices[i], j, surface, &presentationSupport);
                        bool supports_presentation = glfwGetPhysicalDevicePresentationSupport(instance, physical_devices[i], j);
                        int presentation_ok = !desc.require_presentation || supports_presentation;

                        if (presentation_ok) {
                            info.queue_family_index = j;
                            info.queue_timestamp_queries = info.timestamp_period > 0 && properties.limits.timestampComputeAndGraphics && prop.timestampValidBits > 0;
                            if (supports_presentation) {
                                logging::trace("gfx/device", "    picked for presentation");
                            }
                        } else {
                            logging::trace("gfx/device", "    discarded because of no presentation support");
                        }
                    }
                } else if(prop.queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    if(info.compute_queue_family_index == VK_QUEUE_FAMILY_IGNORED) {
                        info.compute_queue_family_index = j;
                        info.compute_queue_timestamp_queries = info.timestamp_period > 0 && properties.limits.timestampComputeAndGraphics && prop.timestampValidBits > 0;
                    }
                } else if(prop.queueFlags & VK_QUEUE_TRANSFER_BIT) {
                    if(info.copy_queue_family_index == VK_QUEUE_FAMILY_IGNORED) {
                        info.copy_queue_family_index = j;
                        info.copy_queue_timestamp_queries = info.timestamp_period > 0 && prop.timestampValidBits > 0;
                    }
                }
            }
        }

        char driver_version[64] = {};
        if (properties.vendorID == 4318) {
            // NVIDIA
            snprintf(driver_version, sizeof(driver_version), "%u.%u.%u.%u",
                (properties.driverVersion >> 22)  & 0x3FF,
                (properties.driverVersion >> 14)  & 0xFF,
                (properties.driverVersion >> 6)  & 0xFF,
                properties.driverVersion & 0x3F
            );
        }
#ifdef _WIN32
        else if (properties.vendorID == 0x8086) {
            // Intel on windows
            snprintf(driver_version, sizeof(driver_version), "%u.%u",
                properties.driverVersion >> 14,
                properties.driverVersion & 0x3FFF
            );
        }
#endif
        else {
            snprintf(driver_version, sizeof(driver_version), "%u.%u.%u",
                VK_API_VERSION_MAJOR(properties.driverVersion), VK_API_VERSION_MINOR(properties.driverVersion), VK_API_VERSION_PATCH(properties.driverVersion)
            );
        }

        logging::info("gfx/device", "Physical device %u:\n    Name: %s\n    Vulkan version: %u.%u.%u\n    Drivers version: %s\n    Vendor ID: %u\n    Device ID: %u\n    Device type: %s\n    QueueFamily: 0x%x\n    ComputeQueueFamily: 0x%x\n    CopyQueueFamily: 0x%x\n",
            i, properties.deviceName,
            VK_API_VERSION_MAJOR(properties.apiVersion), VK_API_VERSION_MINOR(properties.apiVersion), VK_API_VERSION_PATCH(properties.apiVersion),
            driver_version, properties.vendorID, properties.deviceID, string_VkPhysicalDeviceType(properties.deviceType),
            info.queue_family_index, info.compute_queue_family_index, info.copy_queue_family_index);


        u32 extensions_count = 0;
        vkEnumerateDeviceExtensionProperties(physical_devices[i], 0, &extensions_count, 0);

        Array<VkExtensionProperties> extensions(extensions_count);
        vkEnumerateDeviceExtensionProperties(physical_devices[i], 0, &extensions_count, extensions.data);

        // Check if portability extension is required
        for (usize i = 0; i < extensions.length; i++) {
            if (strcmp(extensions[i].extensionName, "VK_KHR_portability_subset") == 0) {
                info.require_portability_subset_extension = true;
                logging::info("gfx/device", "VK_KHR_portability_subset required for device creation");
                break;
            }
        }

        // Check if all features we need are supported.
        if (desc.minimum_api_version >= VK_API_VERSION_1_1 && vkGetPhysicalDeviceFeatures2) {
            VkPhysicalDeviceFeatures2 features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
            features.pNext = sup_next;
            vkGetPhysicalDeviceFeatures2(physical_devices[i], &features);

#define CHECK_FEATURES(o) \
            bool all_features_supported = true; \
            for (usize i = 0; i < ArrayCount(o##_deps.features_sup); i++ ) { \
                if (!CheckAllSupported(*o##_deps.features_req[i], *o##_deps.features_sup[i])) { \
                    all_features_supported = false; \
                    break; \
                } \
            } \

#define CHECK_EXTENSIONS(o) \
            bool all_extensions_supported = true; \
            for (usize i = 0; i < ArrayCount(o##_deps.extensions); i++) { \
                bool found = false; \
                for (usize j = 0; j < extensions.length; j++) { \
                    if (strcmp(o##_deps.extensions[i], extensions[j].extensionName) == 0) { \
                        found = true; \
                        break; \
                    } \
                } \
                if (!found) { \
                    all_extensions_supported = false; \
                    break; \
                } \
            } \

#define CHECK_SUPPORTED_FEATURES_AND_EXTENSIONS(o) \
            if (features_to_check & o##_deps.flag) { \
                CHECK_FEATURES(o) \
                CHECK_EXTENSIONS(o) \
                logging::trace("gfx/device", "%-25s | features: %s, extensions: %s", #o, all_features_supported ? "yes" : " no", all_extensions_supported ? "yes" : " no"); \
                if (all_features_supported && all_extensions_supported) { \
                    info.supported_features = info.supported_features | o##_deps.flag; \
                } \
            }

#define CHECK_SUPPORTED_EXTENSIONS(o) \
            if (features_to_check & o##_deps.flag) { \
                CHECK_EXTENSIONS(o) \
                logging::trace("gfx/device", "%-25s | extensions: %s", #o, all_extensions_supported ? "yes" : " no"); \
                if (all_extensions_supported) { \
                    info.supported_features = info.supported_features | o##_deps.flag; \
                } \
            }

            CHECK_SUPPORTED_FEATURES_AND_EXTENSIONS(dynamic_rendering);
            CHECK_SUPPORTED_FEATURES_AND_EXTENSIONS(synchronization_2);
            CHECK_SUPPORTED_FEATURES_AND_EXTENSIONS(descriptor_indexing);
            CHECK_SUPPORTED_FEATURES_AND_EXTENSIONS(scalar_block_layout);
            CHECK_SUPPORTED_FEATURES_AND_EXTENSIONS(ray_query);
            CHECK_SUPPORTED_FEATURES_AND_EXTENSIONS(ray_tracing_pipeline);
            CHECK_SUPPORTED_EXTENSIONS(external_resources);
            CHECK_SUPPORTED_FEATURES_AND_EXTENSIONS(host_query_reset);
            CHECK_SUPPORTED_EXTENSIONS(calibrated_timestamps);
            CHECK_SUPPORTED_FEATURES_AND_EXTENSIONS(timeline_semaphore);

            if (features_to_check & DeviceFeatures::WIDE_LINES)
                info.supported_features = info.supported_features | DeviceFeatures((features.features.wideLines ? DeviceFeatures::WIDE_LINES : 0));

            logging::trace("gfx/debug", "Supported features: 0x%zx", info.supported_features.flags);

            // We clear the supported flags here. It's not obvious if the spec requires this, but I assume that if
            // a device does not know about a feature struct, it might also not know how large it is and might not
            // be able to clear it.
#define CLEAR(o, flags) \
            if (features_to_check & ((DeviceFeatures)flags)) { \
                ClearFeatures(o##_sup); \
            }

            CLEAR(dynamic_rendering_features, DeviceFeatures::DYNAMIC_RENDERING);
            CLEAR(synchronization_2_features, DeviceFeatures::SYNCHRONIZATION_2);
            CLEAR(scalar_block_layout_features, DeviceFeatures::SCALAR_BLOCK_LAYOUT);
            CLEAR(descriptor_indexing_features, DeviceFeatures::DESCRIPTOR_INDEXING);
            CLEAR(buffer_device_address_features, DeviceFeatures::RAY_TRACING_PIPELINE | DeviceFeatures::RAY_QUERY);
            CLEAR(acceleration_structure_features, DeviceFeatures::RAY_TRACING_PIPELINE | DeviceFeatures::RAY_QUERY);
            CLEAR(ray_query_features, DeviceFeatures::RAY_QUERY);
            CLEAR(ray_tracing_pipeline_features, DeviceFeatures::RAY_TRACING_PIPELINE);
            CLEAR(host_query_reset_features, DeviceFeatures::HOST_QUERY_RESET);
            CLEAR(timeline_semaphore_features, DeviceFeatures::TIMELINE_SEMAPHORES);
        }

        if (properties.apiVersion < desc.minimum_api_version) {
            logging::info("gfx/device", "Discarded because device API version (%u.%u.%u) is below required API version (%u.%u.%u)",
                VK_API_VERSION_MAJOR(properties.apiVersion), VK_API_VERSION_MINOR(properties.apiVersion), VK_API_VERSION_PATCH(properties.apiVersion),
                VK_API_VERSION_MAJOR(desc.minimum_api_version), VK_API_VERSION_MINOR(desc.minimum_api_version), VK_API_VERSION_PATCH(desc.minimum_api_version));
            continue;
        }
        if (info.queue_family_index == VK_QUEUE_FAMILY_IGNORED) {
            logging::info("gfx/device", "Discarded because no suitable queue found");
            continue;
        }
        if ((info.supported_features & desc.required_features) != desc.required_features) {
            logging::info("gfx/device", "Discarded because does not support all required features. Missing features: 0x%zx", (~info.supported_features & desc.required_features).flags);
            continue;
        }

        static const u32 INVALID_PHYSICAL_DEVICE_INDEX = ~0U;
        bool force_device = desc.force_physical_device_index != INVALID_PHYSICAL_DEVICE_INDEX;

        // TODO: always prefer non CPU devices
        bool picked = false;
        if (force_device) {
            if (i == desc.force_physical_device_index) {
                logging::info("gfx/device", "Picked because of forced device index (%u)", i);
                picked = true;
            } else {
                logging::info("gfx/device", "Discarded because forced device index (%u) does not match device index (%u)", desc.force_physical_device_index, i);
            }
        } else {
            if (picked_index == physical_device_count) {
                logging::info("gfx/device", "Picked because first suitable device");
                picked = true;
            } else {
                bool picked_discrete = picked_info.device_type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
                bool picked_integrated = picked_info.device_type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
                bool discrete = info.device_type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
                bool integrated = info.device_type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;

                if (!(picked_discrete || picked_integrated) && (discrete || integrated)) {
                    logging::info("gfx/device", "Picked because first integrated or discrete GPU");
                    picked = true;
                } else {
                    if (desc.prefer_discrete_gpu) {
                        if (!picked_discrete) {
                            if (discrete) {
                                logging::info("gfx/device", "Picked because first discrete GPU");
                                picked = true;
                            } else {
                                logging::info("gfx/device", "Discarded because not a discrete GPU and a suitable device was already found");
                            }
                        } else {
                            logging::info("gfx/device", "Discarded because a discrete GPU was already found");
                        }
                    } else {
                        if (!picked_integrated) {
                            if (integrated) {
                                logging::info("gfx/device", "Picked because first integrated GPU");
                                picked = true;
                            } else {
                                logging::info("gfx/device", "Discarded because not an integrated GPU and a suitable device was already found");
                            }
                        } else {
                            logging::info("gfx/device", "Discarded because an integrated GPU was already found");
                        }
                    }
                }
            }
        }

        if (picked) {
            picked_info = info;
            picked_index = i;
        }
    }

    // Check that a valid device is found.
    if (picked_index == physical_device_count) {
        logging::error("gfx/device", "No valid physical device found");
        return Result::NO_VALID_DEVICE_FOUND;
    } else {
        logging::info("gfx/device", "Picked device: %u", picked_index);
    }

    // Enabled features and extensions
    Array<GenericFeatureStruct*> enabled_features;
    Array<const char*> enabled_extensions;
    if (desc.require_presentation) enabled_extensions.add("VK_KHR_swapchain");
    if (picked_info.require_portability_subset_extension) {
        enabled_extensions.add("VK_KHR_portability_subset");
    }

    // Deduplicate features and extensions. This is O(n^2) without a set, but we don't suport enough features yet to care.

#define ENABLE_FEATURES(o) \
    for (usize i = 0; i < ArrayCount(o##_deps.features_req); i++ ) { \
        if (!enabled_features.contains(o##_deps.features_req[i])) { \
            enabled_features.add(o##_deps.features_req[i]); \
        } \
    } \

#define ENABLE_EXTENSIONS(o) \
    for (usize i = 0; i < ArrayCount(o##_deps.extensions); i++ ) { \
        bool found = false; \
        for (usize j = 0; j < enabled_extensions.length; j++) { \
            if (strcmp(o##_deps.extensions[i], enabled_extensions[j]) == 0) { \
                found = true; \
                break; \
            } \
        } \
        if (!found) { \
            enabled_extensions.add(o##_deps.extensions[i]); \
        } \
    } \

#define ENABLE_FEATURES_AND_EXTENSIONS_IF_SUPPORTED(o) \
    if (picked_info.supported_features & o##_deps.flag) { \
        ENABLE_FEATURES(o) \
        ENABLE_EXTENSIONS(o) \
    }

#define ENABLE_EXTENSIONS_IF_SUPPORTED(o) \
    if (picked_info.supported_features & o##_deps.flag) { \
        ENABLE_EXTENSIONS(o) \
    }

    ENABLE_FEATURES_AND_EXTENSIONS_IF_SUPPORTED(dynamic_rendering);
    ENABLE_FEATURES_AND_EXTENSIONS_IF_SUPPORTED(synchronization_2);
    ENABLE_FEATURES_AND_EXTENSIONS_IF_SUPPORTED(descriptor_indexing);
    ENABLE_FEATURES_AND_EXTENSIONS_IF_SUPPORTED(scalar_block_layout);
    ENABLE_FEATURES_AND_EXTENSIONS_IF_SUPPORTED(ray_query);
    ENABLE_FEATURES_AND_EXTENSIONS_IF_SUPPORTED(ray_tracing_pipeline);
    ENABLE_EXTENSIONS_IF_SUPPORTED(external_resources);
    ENABLE_FEATURES_AND_EXTENSIONS_IF_SUPPORTED(host_query_reset);
    ENABLE_EXTENSIONS_IF_SUPPORTED(calibrated_timestamps);
    ENABLE_FEATURES_AND_EXTENSIONS_IF_SUPPORTED(timeline_semaphore);

    void* enabled_next = 0;
    for (usize i = 0; i < enabled_features.length; i++) {
        Chain(&enabled_next, enabled_features[i]);
    }

    VkPhysicalDeviceFeatures enabled_physical_device_features = {};
    if (picked_info.supported_features & DeviceFeatures::WIDE_LINES) {
        enabled_physical_device_features.wideLines = true;
    }

    // Create a physical device.
    VkDevice device = 0;

    float queue_priorities[] = { 1.0f };
    ArrayFixed<VkDeviceQueueCreateInfo, 3> queue_create_info(1);
    queue_create_info[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info[0].queueFamilyIndex = picked_info.queue_family_index;
    queue_create_info[0].queueCount = 1;
    queue_create_info[0].pQueuePriorities = queue_priorities;

    if (picked_info.compute_queue_family_index != VK_QUEUE_FAMILY_IGNORED) {
        queue_create_info.add({
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = picked_info.compute_queue_family_index,
            .queueCount = 1,
            .pQueuePriorities = queue_priorities,
        });
    }

    if (picked_info.copy_queue_family_index != VK_QUEUE_FAMILY_IGNORED) {
        queue_create_info.add({
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = picked_info.copy_queue_family_index,
            .queueCount = 1,
            .pQueuePriorities = queue_priorities,
        });
    }

    VkDeviceCreateInfo device_create_info = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    device_create_info.pNext = enabled_next;
    device_create_info.queueCreateInfoCount = queue_create_info.length;
    device_create_info.pQueueCreateInfos = queue_create_info.data;
    device_create_info.enabledExtensionCount = (u32)enabled_extensions.length;
    device_create_info.ppEnabledExtensionNames = enabled_extensions.data;
    device_create_info.pEnabledFeatures = &enabled_physical_device_features;

    result = vkCreateDevice(physical_devices[picked_index], &device_create_info, 0, &device);
    if (result != VK_SUCCESS) {
        return Result::DEVICE_CREATION_FAILED;
    }

    // FEATURE: enable VMA extensions if supported
    VmaAllocatorCreateInfo vma_info = {};
    vma_info.flags = desc.required_features & (DeviceFeatures::RAY_QUERY | DeviceFeatures::RAY_TRACING_PIPELINE) ? VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT : 0; // Optionally set here that we externally synchronize.
    vma_info.instance = instance;
    vma_info.physicalDevice = physical_devices[picked_index];
    vma_info.device = device;
    vma_info.vulkanApiVersion = desc.minimum_api_version;

    VmaAllocator vma;
    result = vmaCreateAllocator(&vma_info, &vma);
    if (result != VK_SUCCESS) {
        return Result::VMA_CREATION_FAILED;
    }

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, picked_info.queue_family_index, 0, &queue);

    VkQueue compute_queue = VK_NULL_HANDLE;
    if (picked_info.compute_queue_family_index != VK_QUEUE_FAMILY_IGNORED) {
        vkGetDeviceQueue(device, picked_info.compute_queue_family_index, 0, &compute_queue);
    }

    VkQueue copy_queue = VK_NULL_HANDLE;
    if (picked_info.copy_queue_family_index != VK_QUEUE_FAMILY_IGNORED) {
        vkGetDeviceQueue(device, picked_info.copy_queue_family_index, 0, &copy_queue);
    }

    // Create sync command
    VkCommandPoolCreateInfo sync_pool_info = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    sync_pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;// | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    sync_pool_info.queueFamilyIndex = picked_info.queue_family_index;

    VkCommandPool sync_command_pool = {};
    result = vkCreateCommandPool(device, &sync_pool_info, 0, &sync_command_pool);
    if (result != VK_SUCCESS) {
        return Result::API_OUT_OF_MEMORY;
    }

    VkCommandBufferAllocateInfo sync_allocate_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    sync_allocate_info.commandPool = sync_command_pool;
    sync_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    sync_allocate_info.commandBufferCount = 1;

    VkCommandBuffer sync_command_buffer = {};
    result = vkAllocateCommandBuffers(device, &sync_allocate_info, &sync_command_buffer);
    if (result != VK_SUCCESS) {
        return Result::API_OUT_OF_MEMORY;
    }

    VkFence sync_fence = {};
    VkFenceCreateInfo fence_info = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    result = vkCreateFence(device, &fence_info, 0, &sync_fence);
    if (result != VK_SUCCESS) {
        return Result::API_OUT_OF_MEMORY;
    }

    if (debug_utils_enabled) {
        VkDebugUtilsObjectNameInfoEXT name_info = { VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT }; \

        DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_QUEUE, queue, "xpg-queue");
        DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_QUEUE, compute_queue, "xpg-compute-queue");
        DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_QUEUE, copy_queue, "xpg-transfer-queue");
        DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_COMMAND_BUFFER, sync_command_buffer, "xpg-sync-command-buffer");
        DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_COMMAND_POOL, sync_command_pool, "xpg-sync-command-pool");
        DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_FENCE, sync_fence, "xpg-sync-fence");
    }

    vk->instance_version = instance_version;
    vk->device_version = picked_info.device_api_version;
    vk->instance = instance;
    vk->physical_device = physical_devices[picked_index];
    vk->device = device;
    vk->device_features = picked_info.supported_features;
    vk->timestamp_period_ns = picked_info.timestamp_period;
    vk->queue = queue;
    vk->queue_family_index = picked_info.queue_family_index;
    vk->queue_timestamp_queries = picked_info.queue_timestamp_queries;
    vk->compute_queue = compute_queue;
    vk->compute_queue_family_index = picked_info.compute_queue_family_index;
    vk->compute_queue_timestamp_queries = picked_info.compute_queue_timestamp_queries;
    vk->copy_queue = copy_queue;
    vk->copy_queue_family_index = picked_info.copy_queue_family_index;
    vk->copy_queue_timestamp_queries = picked_info.copy_queue_timestamp_queries;
    vk->preferred_frames_in_flight = desc.preferred_frames_in_flight;
    vk->vsync = desc.vsync;
    vk->vma = vma;
    vk->sync_command_pool = sync_command_pool;
    vk->sync_command_buffer = sync_command_buffer;
    vk->sync_fence = sync_fence;
    vk->debug_utils_enabled = debug_utils_enabled;
    vk->debug_messenger = debug_messenger;

    return Result::SUCCESS;
}

void DestroyContext(Context* vk) {
    vkFreeCommandBuffers(vk->device, vk->sync_command_pool, 1, &vk->sync_command_buffer);
    vkDestroyCommandPool(vk->device, vk->sync_command_pool, 0);
    vkDestroyFence(vk->device, vk->sync_fence, 0);

    // VMA
    vmaDestroyAllocator(vk->vma);

    // Device and instance
    vkDestroyDevice(vk->device, 0);
    if (vk->debug_messenger) {
        vkDestroyDebugUtilsMessengerEXT(vk->instance, vk->debug_messenger, 0);
    }
    vkDestroyInstance(vk->instance, 0);

    *vk = {};
}

void WaitIdle(Context& vk) {
    vkDeviceWaitIdle(vk.device);
}

Result
CreateSwapchain(Window* w, const Context& vk, VkSurfaceKHR surface, VkFormat format, u32 fb_width, u32 fb_height, usize frames, VkPresentModeKHR present_mode, VkSwapchainKHR old_swapchain)
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
    swapchain_info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    swapchain_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchain_info.queueFamilyIndexCount = 1;
    swapchain_info.pQueueFamilyIndices = queue_family_indices;
    swapchain_info.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    swapchain_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    // NOTE: About smooth resize:
    // - On windows it looks like with IMMEDIATE and MAILBOX then we have no artifacts.
    // - Perfect solution would be to use WM_ENTERSIZEMOVE and WM_EXITSIZEMOVE
    //   to change the presentation mode. However you then need to do vsync
    //   in some other way if you don't want to run at 500 fps during resize.
    //   This likely entails calling into DirectComposition / DWM stuff.
    // - Maybe can also try in software to never queue more than one frame? Does not seem to make any difference
    // - Still need to check on linux
    // - Also no sync resize could affect this, maybe easier to implement?
    //
    // See:
    // - Raph levien blog: https://raphlinus.github.io/rust/gui/2019/06/21/smooth-resize-test.html
    // - Winit discussion: https://github.com/rust-windowing/winit/issues/786
    // swapchain_info.presentMode = vk.vsync ? VK_PRESENT_MODE_FIFO_KHR : VK_PRESENT_MODE_MAILBOX_KHR;
    swapchain_info.presentMode = present_mode;
    swapchain_info.oldSwapchain = old_swapchain;

    VkSwapchainKHR swapchain;
    VkResult result = vkCreateSwapchainKHR(vk.device, &swapchain_info, 0, &swapchain);
    if (result != VK_SUCCESS) {
	logging::error("gfx/swapchain", "vkCreateSwapchainKHR [%ux%u] failed: %d", fb_width, fb_height, result);
        return Result::SWAPCHAIN_CREATION_FAILED;
    }

    // Get swapchain images.
    u32 image_count;
    result = vkGetSwapchainImagesKHR(vk.device, swapchain, &image_count, 0);
    if (result != VK_SUCCESS) {
	logging::error("gfx/swapchain", "vkGetSwapchainImages for count failed: %d", result);
        return Result::API_OUT_OF_MEMORY;
    }

    Array<VkImage> images(image_count);
    result = vkGetSwapchainImagesKHR(vk.device, swapchain, &image_count, images.data);
    if (result != VK_SUCCESS) {
	logging::error("gfx/swapchain", "vkGetSwapchainImages for values failed: %d", result);
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
            logging::error("gfx/swapchain", "vkCreateImageView for image %zu failed: %d", i, result);
            return Result::API_OUT_OF_MEMORY;
        }

        if (vk.debug_utils_enabled) {
            VkDebugUtilsObjectNameInfoEXT name_info = { VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT }; \
            VkDevice device = vk.device;
            DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_IMAGE, images[i], "xpg-swapchain-image");
            DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_IMAGE_VIEW, image_views[i], "xpg-swapchain-image-view");
        }
    }

    w->swapchain = swapchain;
    w->images = move(images);
    w->image_views = move(image_views);
    w->fb_width = fb_width;
    w->fb_height = fb_height;
    w->present_mode = present_mode;

    return Result::SUCCESS;
}

SwapchainStatus UpdateSwapchain(Window* w, const Context& vk)
{
    VkSurfaceCapabilitiesKHR surface_capabilities;
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.physical_device, w->surface, &surface_capabilities) != VK_SUCCESS) {
        return SwapchainStatus::FAILED;
    }

    uint32_t new_width = surface_capabilities.currentExtent.width == 0xFFFFFFFF ? w->fb_width : surface_capabilities.currentExtent.width;
    uint32_t new_height = surface_capabilities.currentExtent.height == 0xFFFFFFFF ? w->fb_height : surface_capabilities.currentExtent.height;

    if (new_width == 0 || new_height == 0)
        return SwapchainStatus::MINIMIZED;

    if (!w->force_swapchain_recreate && (new_width == w->fb_width && new_height == w->fb_height)) {
        return SwapchainStatus::READY;
    }

    VkSwapchainKHR old_swapchain = w->swapchain;
#if SYNC_SWAPCHAIN_DESTRUCTION
    vkDeviceWaitIdle(vk.device);

    for (size_t i = 0; i < w->image_views.length; i++) {
        vkDestroyImageView(vk.device, w->image_views[i], nullptr);
        w->image_views[i] = VK_NULL_HANDLE;
    }
#else
    w->stale_swapchains.add(StaleSwapchain {
        .frames_in_flight = w->frames.length,
        .swapchain = old_swapchain,
        .image_views = std::move(w->image_views),
    });
    logging::trace("gfx/swapchain", "Added stale swapchain. Total: %llu", w->stale_swapchains.length);
#endif

    Result result = CreateSwapchain(w, vk, w->surface, w->swapchain_format, new_width, new_height, w->images.length, w->present_mode, old_swapchain);
    if (result != Result::SUCCESS) {
        return SwapchainStatus::FAILED;
    }

    w->force_swapchain_recreate = false;

#if SYNC_SWAPCHAIN_DESTRUCTION
    vkDestroySwapchainKHR(vk.device, old_swapchain, nullptr);
#endif

    return SwapchainStatus::RESIZED;
}

Frame& WaitForFrame(Window* w, const Context& vk) {
    Frame& frame = w->frames[w->swapchain_frame_index];
    vkWaitForFences(vk.device, 1, &frame.fence, VK_TRUE, ~0ULL);
    vkResetFences(vk.device, 1, &frame.fence);

#if !SYNC_SWAPCHAIN_DESTRUCTION
    // Decrement frame in flight count on stale swapchains, if any.
    for (usize i = 0; i < w->stale_swapchains.length;) {
        if(--w->stale_swapchains[i].frames_in_flight == 0) {
            StaleSwapchain& stale = w->stale_swapchains[i];
            vkDestroySwapchainKHR(vk.device, stale.swapchain, nullptr);
            for (size_t i = 0; i < stale.image_views.length; i++) {
                vkDestroyImageView(vk.device, stale.image_views[i], nullptr);
            }

            // Replace with last element from array, if any
            if(i < w->stale_swapchains.length - 1) {
                w->stale_swapchains[i] = move(w->stale_swapchains[w->stale_swapchains.length - 1]);
            }

            // Pop last element
            w->stale_swapchains.length -= 1;
        } else {
            i += 1;
        }
    }
#endif

    return frame;
}

Result AcquireImage(Frame* frame, Window* window, const Context& vk)
{
    u32 image_index;
    VkResult vkr = vkAcquireNextImageKHR(vk.device, window->swapchain, ~0ull, frame->acquire_semaphore, 0, &image_index);
    if (vkr == VK_ERROR_OUT_OF_DATE_KHR) {
        window->force_swapchain_recreate = true;
        return Result::SWAPCHAIN_OUT_OF_DATE;
    }
    else if (vkr != VK_SUCCESS) {
        return Result::API_ERROR;
    }

    frame->current_image_index = image_index;
    frame->current_image = window->images[image_index];
    frame->current_image_view = window->image_views[image_index];

    return Result::SUCCESS;
}

// @API(frame): Deprecated in favor of split acquire / wait?
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

VkResult
BeginCommands(VkCommandPool pool, VkCommandBuffer buffer, const Context& vk) {

    // Reset command pool
    VkResult vkr;
    vkr = vkResetCommandPool(vk.device, pool, 0);
    if (vkr != VK_SUCCESS) {
        return vkr;
    }

    // Record commands
    VkCommandBufferBeginInfo begin_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkr = vkBeginCommandBuffer(buffer, &begin_info);
    if (vkr != VK_SUCCESS) {
        return vkr;
    }

    return VK_SUCCESS;
}

VkResult
EndCommands(VkCommandBuffer buffer) {
    return vkEndCommandBuffer(buffer);
}

VkResult CreateQueryPool(VkQueryPool* pool, const Context& vk, const QueryPoolDesc&& desc) {
    VkQueryPoolCreateInfo query_pool_info = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    query_pool_info.queryType = desc.type;
    query_pool_info.queryCount = desc.count;
    return vkCreateQueryPool(vk.device, &query_pool_info, NULL, pool);
}

void DestroyQueryPool(VkQueryPool* pool, const Context& vk) {
    vkDestroyQueryPool(vk.device, *pool, NULL);
    *pool = VK_NULL_HANDLE;
}

VkResult SubmitQueue(VkQueue queue, const SubmitDesc&& desc) {
    assert(desc.wait_semaphores.length == desc.wait_stages.length);

    VkSubmitInfo submit_info = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit_info.commandBufferCount = (u32)desc.cmd.length;
    submit_info.pCommandBuffers = desc.cmd.data;
    submit_info.waitSemaphoreCount = (u32)desc.wait_semaphores.length;
    submit_info.pWaitSemaphores = desc.wait_semaphores.data;
    submit_info.pWaitDstStageMask = desc.wait_stages.data;
    submit_info.signalSemaphoreCount = desc.signal_semaphores.length;
    submit_info.pSignalSemaphores = desc.signal_semaphores.data;

    VkTimelineSemaphoreSubmitInfoKHR timeline_submit_info = { VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR };
    if (desc.wait_timeline_values.length || desc.signal_timeline_values.length) {
        timeline_submit_info.waitSemaphoreValueCount = desc.wait_timeline_values.length;
        timeline_submit_info.pWaitSemaphoreValues = desc.wait_timeline_values.data;
        timeline_submit_info.signalSemaphoreValueCount = desc.signal_timeline_values.length;
        timeline_submit_info.pSignalSemaphoreValues = desc.signal_timeline_values.data;

        submit_info.pNext = &timeline_submit_info;
    }

    return vkQueueSubmit(queue, 1, &submit_info, desc.fence);
}

VkResult Submit(const Frame& frame, const Context& vk,
// VkPipelineStageFlags2 stage_mask) {
VkPipelineStageFlags stage_mask) {

#if 0
    VkSemaphoreSubmitInfo wait_info = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
    wait_info.semaphore = frame.acquire_semaphore;
    wait_info.stageMask = stage_mask;

    VkSemaphoreSubmitInfo signal_info = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
    signal_info.semaphore = frame.release_semaphore;
    signal_info.stageMask = stage_mask;

    VkCommandBufferSubmitInfo command_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
    command_info.commandBuffer = frame.command_buffer;

    VkSubmitInfo2 submit_info = { VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    submit_info.pCommandBufferInfos = &command_info;
    submit_info.commandBufferInfoCount = 1;
    submit_info.pWaitSemaphoreInfos = &wait_info;
    submit_info.waitSemaphoreInfoCount = 1;
    submit_info.pSignalSemaphoreInfos = &signal_info;
    submit_info.signalSemaphoreInfoCount = 1;

    return vkQueueSubmit2KHR(vk.queue, 1, &submit_info, frame.fence);
#else
    return SubmitQueue(vk.queue, {
        .cmd = { frame.command_buffer },
        .wait_semaphores = { frame.acquire_semaphore },
        .wait_stages = { stage_mask },
        .signal_semaphores = { frame.release_semaphore },
        .fence = frame.fence,
    });
#endif
}

VkResult SubmitSync(const Context& vk) {
    VkResult vkr = SubmitQueue(vk.queue, {
        .cmd = { vk.sync_command_buffer},
        .fence = vk.sync_fence,
    });
    if (vkr != VK_SUCCESS) {
        return vkr;
    }

    vkWaitForFences(vk.device, 1, &vk.sync_fence, VK_TRUE, ~0);
    vkResetFences(vk.device, 1, &vk.sync_fence);
    return VK_SUCCESS;
}


VkResult PresentFrame(Window* w, Frame* frame, const Context& vk) {
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
        return vkr;
    }

    w->swapchain_frame_index = (w->swapchain_frame_index + 1) % w->frames.length;
    frame->current_image = VK_NULL_HANDLE;
    frame->current_image_view = VK_NULL_HANDLE;
    frame->current_image_index = ~(u32)0;
    return VK_SUCCESS;
}

Result
CreateWindowWithSwapchain(Window* w, const Context& vk, const char* name, u32 width, u32 height, u32 x, u32 y)
{
    // Create window.
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_POSITION_X, x);
    glfwWindowHint(GLFW_POSITION_Y, y);
    GLFWwindow* window = glfwCreateWindow(width, height, name, NULL, NULL);
    if (!window) {
        logging::error("gfx/window", "Failed to create GLFW window");
        return Result::WINDOW_CREATION_FAILED;
    }

    // Create window surface.
    VkSurfaceKHR surface = 0;
    VkResult result = glfwCreateWindowSurface(vk.instance, window, NULL, &surface);
    if (result != VK_SUCCESS) {
        logging::error("gfx/window", "Failed to create GLFW window surface %d", result);
        return Result::SURFACE_CREATION_FAILED;
    }

    // Retrieve surface capabilities.
    VkSurfaceCapabilitiesKHR surface_capabilities = {};
    result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.physical_device, surface, &surface_capabilities);
    if (result != VK_SUCCESS) {
        logging::error("gfx/window", "vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed: %d", result);
        return Result::SWAPCHAIN_CREATION_FAILED;
    }

    // Compute number of frames in flight.
    u32 num_frames = Max<u32>(vk.preferred_frames_in_flight, surface_capabilities.minImageCount);
    if (surface_capabilities.maxImageCount > 0) {
        num_frames = Min<u32>(num_frames, surface_capabilities.maxImageCount);
    }
    logging::info("gfx/window", "Swapchain Frames: preferred %d, min: %d, max %d, picked: %d", vk.preferred_frames_in_flight,
	          surface_capabilities.minImageCount, surface_capabilities.maxImageCount, num_frames);

    // Retrieve supported surface formats.
    // FEATURE: smarter format picking logic (HDR / non sRGB displays).
    u32 formats_count;
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(vk.physical_device, surface, &formats_count, 0);
    if (result != VK_SUCCESS || formats_count == 0) {
        logging::error("gfx/window", "vkGetPhysicalDeviceSurfaceFormatsKHR for count failed: %d", result);
        return Result::SWAPCHAIN_CREATION_FAILED;
    }

    Array<VkSurfaceFormatKHR> formats(formats_count);
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(vk.physical_device, surface, &formats_count, formats.data);
    if (result != VK_SUCCESS || formats_count == 0) {
        logging::error("gfx/window", "vkGetPhysicalDeviceSurfaceFormatsKHR for values failed: %d", result);
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

    logging::info("gfx/window", "Swapchain format: %s", string_VkFormat(format));

    // Retrieve framebuffer size.
    u32 fb_width = surface_capabilities.currentExtent.width == 0xFFFFFFFF ? width : surface_capabilities.currentExtent.width;
    u32 fb_height = surface_capabilities.currentExtent.height == 0xFFFFFFFF ? height : surface_capabilities.currentExtent.height;

    logging::info("gfx/window", "Surface extents: [%ux%u]", fb_width, fb_height);

    // Default to FIFO, this is always supported.
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;

    // If vsync is disabled check if MAILBOX or IMMEDIATE is supported.
    if (!vk.vsync) {
        u32 supported_present_mode_count = 0;
        result = vkGetPhysicalDeviceSurfacePresentModesKHR(vk.physical_device, surface, &supported_present_mode_count, 0);
        if (result != VK_SUCCESS) {
            logging::warning("gfx/window", "first vkGetPhysicalDeviceSurfacePresentModesKHR failed: %d", result);
        } else {
            Array<VkPresentModeKHR> supported_present_modes(supported_present_mode_count);
            result = vkGetPhysicalDeviceSurfacePresentModesKHR(vk.physical_device, surface, &supported_present_mode_count, supported_present_modes.data);
            if (result != VK_SUCCESS) {
                logging::warning("gfx/window", "second vkGetPhysicalDeviceSurfacePresentModesKHR failed: %d", result);
            } else {
                if (supported_present_modes.contains(VK_PRESENT_MODE_MAILBOX_KHR)) {
                    present_mode = VK_PRESENT_MODE_MAILBOX_KHR;
                } else if (supported_present_modes.contains(VK_PRESENT_MODE_IMMEDIATE_KHR)) {
                    present_mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
                }
            }
        }
    }
    logging::info("gfx/window", "Swapchain present mode: %s", string_VkPresentModeKHR(present_mode));

    Result res = CreateSwapchain(w, vk, surface, format, fb_width, fb_height, num_frames, present_mode, VK_NULL_HANDLE);
    if (res != Result::SUCCESS) {
        logging::error("gfx/window", "CreateSwapchain failed: %d", (s32)res);
        return res;
    }

    // Create frames
    Array<Frame> frames(num_frames);
    for (u32 i = 0; i < num_frames; i++) {
        gfx::Frame& frame = frames[i];

        // Graphics commands
        {
            VkCommandPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
            pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
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
        }

        // Async compute commands, if queue is available
        if (vk.compute_queue != VK_NULL_HANDLE) {
            VkCommandPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
            pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
            pool_info.queueFamilyIndex = vk.compute_queue_family_index;

            result = vkCreateCommandPool(vk.device, &pool_info, 0, &frame.compute_command_pool);
            if (result != VK_SUCCESS) {
                return Result::API_OUT_OF_MEMORY;
            }

            VkCommandBufferAllocateInfo allocate_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            allocate_info.commandPool = frame.compute_command_pool;
            allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocate_info.commandBufferCount = 1;

            result = vkAllocateCommandBuffers(vk.device, &allocate_info, &frame.compute_command_buffer);
            if (result != VK_SUCCESS) {
                return Result::API_OUT_OF_MEMORY;
            }
        }

        // Copy commands, if queue is available
        if (vk.copy_queue != VK_NULL_HANDLE) {
            VkCommandPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
            pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
            pool_info.queueFamilyIndex = vk.copy_queue_family_index;

            result = vkCreateCommandPool(vk.device, &pool_info, 0, &frame.copy_command_pool);
            if (result != VK_SUCCESS) {
                return Result::API_OUT_OF_MEMORY;
            }

            VkCommandBufferAllocateInfo allocate_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            allocate_info.commandPool = frame.copy_command_pool;
            allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocate_info.commandBufferCount = 1;

            result = vkAllocateCommandBuffers(vk.device, &allocate_info, &frame.copy_command_buffer);
            if (result != VK_SUCCESS) {
                return Result::API_OUT_OF_MEMORY;
            }
        }

        VkFenceCreateInfo fence_info = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(vk.device, &fence_info, 0, &frame.fence);

        CreateGPUSemaphore(vk.device, &frame.acquire_semaphore);
        CreateGPUSemaphore(vk.device, &frame.release_semaphore);

        if (vk.debug_utils_enabled) {
            VkDebugUtilsObjectNameInfoEXT name_info = { VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT }; \
            VkDevice device = vk.device;

            // Commands
            DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_COMMAND_BUFFER, frame.command_buffer, "xpg-frame-command-buffer");
            DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_COMMAND_POOL,   frame.command_pool,    "xpg-frame-command-pool");
            if (vk.compute_queue != VK_NULL_HANDLE) {
                DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_COMMAND_BUFFER, frame.compute_command_buffer, "xpg-frame-compute-command-buffer");
                DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_COMMAND_POOL,   frame.compute_command_pool,    "xpg-frame-compute-command-pool");
            }
            if (vk.copy_queue != VK_NULL_HANDLE) {
                DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_COMMAND_BUFFER, frame.copy_command_buffer,    "xpg-frame-transfer-command-buffer");
                DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_COMMAND_POOL,   frame.copy_command_pool,    "xpg-frame-transfer-command-pool");
            }

            // Sync
            DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_SEMAPHORE, frame.acquire_semaphore, "xpg-frame-acquire-semaphore");
            DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_SEMAPHORE, frame.release_semaphore, "xpg-frame-release-semaphore");
            DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_FENCE,     frame.fence,             "xpg-frame-fence");
        }
    }

    w->window = window;
    w->surface = surface;
    w->swapchain_format = format;
    w->frames = move(frames);

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

        if(vk.compute_queue) {
            vkFreeCommandBuffers(vk.device, frame.compute_command_pool, 1, &frame.compute_command_buffer);
            vkDestroyCommandPool(vk.device, frame.compute_command_pool, 0);
        }

        if(vk.copy_queue) {
            vkFreeCommandBuffers(vk.device, frame.copy_command_pool, 1, &frame.copy_command_buffer);
            vkDestroyCommandPool(vk.device, frame.copy_command_pool, 0);
        }
    }

    *w = {};
}

void CmdMemoryBarrier(VkCommandBuffer cmd, const MemoryBarrierDesc &&desc)
{
    VkMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    barrier.srcAccessMask = desc.src_access;
    barrier.dstAccessMask = desc.dst_access;
    barrier.srcStageMask = desc.src_stage;
    barrier.dstStageMask = desc.dst_stage;

    VkDependencyInfo info = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    info.memoryBarrierCount= 1;
    info.pMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2KHR(cmd, &info);
}

void CmdBufferBarrier(VkCommandBuffer cmd, const BufferBarrierDesc&& desc)
{
    VkBufferMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
    barrier.srcAccessMask = desc.src_access;
    barrier.dstAccessMask = desc.dst_access;
    barrier.srcStageMask = desc.src_stage;
    barrier.dstStageMask = desc.dst_stage;
    barrier.srcQueueFamilyIndex = desc.src_queue;
    barrier.dstQueueFamilyIndex = desc.dst_queue;
    barrier.buffer = desc.buffer;
    barrier.offset = desc.offset;
    barrier.size = desc.size;

    VkDependencyInfo info = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    info.bufferMemoryBarrierCount= 1;
    info.pBufferMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2KHR(cmd, &info);
}

void
CmdImageBarrier(VkCommandBuffer cmd, const ImageBarrierDesc&& desc)
{
    VkImageMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = desc.image;
    barrier.subresourceRange.aspectMask = desc.aspect_mask;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = desc.src_access;
    barrier.dstAccessMask = desc.dst_access;
    barrier.srcStageMask = desc.src_stage;
    barrier.dstStageMask = desc.dst_stage;
    barrier.oldLayout = desc.old_layout;
    barrier.newLayout = desc.new_layout;

    VkDependencyInfo info = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    info.imageMemoryBarrierCount = 1;
    info.pImageMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2KHR(cmd, &info);
}

void CmdBarriers(VkCommandBuffer cmd, const BarriersDesc &&desc)
{
    VkDependencyInfo info = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    info.memoryBarrierCount= desc.memory.length;
    info.pMemoryBarriers = (VkMemoryBarrier2*)desc.memory.data;
    info.bufferMemoryBarrierCount = desc.buffer.length;
    info.pBufferMemoryBarriers = (VkBufferMemoryBarrier2*)desc.buffer.data;
    info.imageMemoryBarrierCount = desc.image.length;
    info.pImageMemoryBarriers = (VkImageMemoryBarrier2*)desc.image.data;

    vkCmdPipelineBarrier2KHR(cmd, &info);
}

void CmdBeginRendering(VkCommandBuffer cmd, const BeginRenderingDesc&& desc)
{
    VkRenderingAttachmentInfo depth { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    depth.imageView = desc.depth.view;
    depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depth.resolveMode = VK_RESOLVE_MODE_NONE;
    depth.loadOp = desc.depth.load_op;
    depth.storeOp = desc.depth.store_op;
    depth.clearValue.depthStencil.depth = desc.depth.clear;

    VkRenderingInfo rendering_info = { VK_STRUCTURE_TYPE_RENDERING_INFO };
    rendering_info.renderArea.offset.x = desc.offset_x;
    rendering_info.renderArea.offset.y = desc.offset_y;
    rendering_info.renderArea.extent.width = desc.width;
    rendering_info.renderArea.extent.height = desc.height;
    rendering_info.layerCount = 1;
    rendering_info.viewMask = 0;
    rendering_info.colorAttachmentCount = desc.color.length;
    rendering_info.pColorAttachments = (VkRenderingAttachmentInfo*)desc.color.data;
    rendering_info.pDepthAttachment = 0;
    rendering_info.pStencilAttachment = 0;

    if (depth.imageView) {
        rendering_info.pDepthAttachment = &depth;
    }

    vkCmdBeginRenderingKHR(cmd, &rendering_info);
}

void CmdEndRendering(VkCommandBuffer cmd)
{
    vkCmdEndRenderingKHR(cmd);
}

void CmdCopyBuffer(VkCommandBuffer cmd, const CopyBufferDesc&& desc)
{
    VkBufferCopy region = {};
    region.srcOffset = desc.src_offset;
    region.dstOffset = desc.dst_offset;
    region.size = desc.size;
    vkCmdCopyBuffer(cmd, desc.src, desc.dst, 1, &region);
}

void CmdCopyImageToBuffer(VkCommandBuffer cmd, const CopyImageBufferDesc&& desc)
{
    VkBufferImageCopy region = {};
    region.bufferOffset = desc.buffer_offset;
    region.bufferRowLength = desc.buffer_row_stride;
    region.bufferImageHeight = desc.buffer_image_height;
    region.imageSubresource.aspectMask = desc.image_aspect;
    region.imageSubresource.mipLevel = desc.image_mip;
    region.imageSubresource.baseArrayLayer = desc.image_base_layer;
    region.imageSubresource.layerCount = desc.image_layer_count;
    region.imageOffset.x = desc.image_x;
    region.imageOffset.y = desc.image_y;
    region.imageOffset.z = desc.image_z;
    region.imageExtent.width = desc.image_width;
    region.imageExtent.height = desc.image_height;
    region.imageExtent.depth = desc.image_depth;
    vkCmdCopyImageToBuffer(cmd, desc.image, desc.image_layout, desc.buffer, 1, &region);
}

void CmdCopyBufferToImage(VkCommandBuffer cmd, const CopyImageBufferDesc&& desc)
{
    VkBufferImageCopy region = {};
    region.bufferOffset = desc.buffer_offset;
    region.bufferRowLength = desc.buffer_row_stride;
    region.bufferImageHeight = desc.buffer_image_height;
    region.imageSubresource.aspectMask = desc.image_aspect;
    region.imageSubresource.mipLevel = desc.image_mip;
    region.imageSubresource.baseArrayLayer = desc.image_base_layer;
    region.imageSubresource.layerCount = desc.image_layer_count;
    region.imageOffset.x = desc.image_x;
    region.imageOffset.y = desc.image_y;
    region.imageOffset.z = desc.image_z;
    region.imageExtent.width = desc.image_width;
    region.imageExtent.height = desc.image_height;
    region.imageExtent.depth = desc.image_depth;
    vkCmdCopyBufferToImage(cmd, desc.buffer, desc.image, desc.image_layout, 1, &region);
}

#ifdef _WIN32
VkExternalSemaphoreHandleTypeFlagBits EXTERNAL_SEMAPHORE_HANDLE_TYPE_BIT = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
VkExternalMemoryHandleTypeFlagBits EXTERNAL_MEMORY_HANDLE_TYPE_BIT = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
VkExternalSemaphoreHandleTypeFlagBits EXTERNAL_SEMAPHORE_HANDLE_TYPE_BIT = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
VkExternalMemoryHandleTypeFlagBits EXTERNAL_MEMORY_HANDLE_TYPE_BIT = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

VkResult
CreateGPUSemaphore(VkDevice device, VkSemaphore* semaphore, bool external)
{
    VkSemaphoreCreateInfo semaphore_info = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };

    VkExportSemaphoreCreateInfo export_semaphore_info = { VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO };
    export_semaphore_info.handleTypes = EXTERNAL_SEMAPHORE_HANDLE_TYPE_BIT;

    if(external) {
        semaphore_info.pNext = &export_semaphore_info;
    }

    return vkCreateSemaphore(device, &semaphore_info, 0, semaphore);
}

VkResult
CreateGPUTimelineSemaphore(VkDevice device, VkSemaphore* semaphore, u64 initial_value, bool external)
{
    VkSemaphoreTypeCreateInfo timeline_create_info = { VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
    timeline_create_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timeline_create_info.initialValue = initial_value;

    VkSemaphoreCreateInfo semaphore_info = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    semaphore_info.pNext = &timeline_create_info;

    VkExportSemaphoreCreateInfo export_semaphore_info = { VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO };
    export_semaphore_info.handleTypes = EXTERNAL_SEMAPHORE_HANDLE_TYPE_BIT;

    if(external) {
        timeline_create_info.pNext = &export_semaphore_info;
    }

    return vkCreateSemaphore(device, &semaphore_info, 0, semaphore);
}

void
DestroyGPUSemaphore(VkDevice device, VkSemaphore* semaphore)
{
    vkDestroySemaphore(device, *semaphore, 0);
    *semaphore = VK_NULL_HANDLE;
}


VkResult GetExternalHandleForSemaphore(ExternalHandle* handle, const Context& vk, VkSemaphore semaphore) {
#ifdef XPG_MOLTENVK_STATIC
    return VK_ERROR_FEATURE_NOT_PRESENT;
#else
#if _WIN32
    VkSemaphoreGetWin32HandleInfoKHR semaphore_get_handle_info = { VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR };
    semaphore_get_handle_info.semaphore = semaphore;
    semaphore_get_handle_info.handleType = EXTERNAL_SEMAPHORE_HANDLE_TYPE_BIT;

    return vkGetSemaphoreWin32HandleKHR(vk.device, &semaphore_get_handle_info, handle);
#else
    VkSemaphoreGetFdInfoKHR semaphore_get_handle_info = { VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR };
    semaphore_get_handle_info.semaphore = semaphore;
    semaphore_get_handle_info.handleType = EXTERNAL_SEMAPHORE_HANDLE_TYPE_BIT;

    return vkGetSemaphoreFdKHR(vk.device, &semaphore_get_handle_info, handle);
#endif
#endif
}

VkResult
CreateBuffer(Buffer* buffer, const Context& vk, size_t size, const BufferDesc&& desc) {
    // Alloc buffer
    VkExternalMemoryBufferCreateInfo external_memory_buffer_info = { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO };
    external_memory_buffer_info.handleTypes = EXTERNAL_MEMORY_HANDLE_TYPE_BIT;

    // Alloc buffer
    VkBufferCreateInfo buffer_info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    buffer_info.size = size;
    buffer_info.usage = desc.usage;
    if (desc.external) {
        buffer_info.pNext = &external_memory_buffer_info;
    }

    VmaAllocationCreateInfo alloc_create_info = {};
    alloc_create_info.requiredFlags = desc.alloc.memory_properties_required;
    alloc_create_info.preferredFlags = desc.alloc.memory_properties_preferred;
    alloc_create_info.flags = desc.alloc.vma_flags;
    alloc_create_info.usage = desc.alloc.vma_usage;
    alloc_create_info.pool = desc.pool;

    VkBuffer buf = 0;
    VmaAllocation allocation = {};
    VmaAllocationInfo alloc_info = {};
    VkResult vkr = vmaCreateBuffer(vk.vma, &buffer_info, &alloc_create_info, &buf, &allocation, &alloc_info);
    if (vkr != VK_SUCCESS) {
        return vkr;
    }

    buffer->buffer = buf;
    buffer->allocation = allocation;
    buffer->map = ArrayView<u8>((u8*)alloc_info.pMappedData, size);

    return VK_SUCCESS;
}

VkResult
CreateBufferFromData(Buffer* buffer, const Context& vk, ArrayView<u8> data, const BufferDesc&& desc) {
    VkResult vkr = CreateBuffer(buffer, vk, data.length, move(desc));
    if (vkr != VK_SUCCESS) {
        return vkr;
    }

    // If allocation is already persistenly mapped, just do the memcpy
    if (buffer->map.data) {
        buffer->map.copy_exact(data);
    } else {
        // Check if allocation can be mapped
        VkMemoryPropertyFlags memPropFlags;
        vmaGetAllocationMemoryProperties(vk.vma, buffer->allocation, &memPropFlags);
        if(memPropFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            // Let VMA do the map, copy, unmap steps
            vkr = vmaCopyMemoryToAllocation(vk.vma, data.data, buffer->allocation, 0, data.length);
            if (vkr != VK_SUCCESS) {
                return vkr;
            }
        } else {
            // Allocate a staging buffer and do the copy through commands
            Buffer staging = {};
            vkr = CreateBuffer(&staging, vk, data.length, {
                .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                .alloc = AllocPresets::HostWriteCombining,
            });

            if (vkr != VK_SUCCESS) {
                DestroyBuffer(buffer, vk);
                return vkr;
            }

            // Copy into staging buffer
            staging.map.copy_exact(data);

            // Upload
            BeginCommands(vk.sync_command_pool, vk.sync_command_buffer, vk);
            CmdCopyBuffer(vk.sync_command_buffer, {
                .src = staging.buffer,
                .dst = buffer->buffer,
                .size = data.length,
            });
            EndCommands(vk.sync_command_buffer);
            SubmitSync(vk);

            // Free staging buffer
            DestroyBuffer(&staging, vk);
        }
    }

    return VK_SUCCESS;
}

void
DestroyBuffer(Buffer* buffer, const Context& vk)
{
    if (buffer->buffer) {
        vmaDestroyBuffer(vk.vma, buffer->buffer, buffer->allocation);
    }
    *buffer = {};
}

VkResult CreatePoolForBuffer(Pool* pool, const Context& vk, const PoolBufferDesc&& desc) {
    // Alloc buffer
    VkExternalMemoryBufferCreateInfo external_memory_buffer_info = { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO };
    external_memory_buffer_info.handleTypes = EXTERNAL_MEMORY_HANDLE_TYPE_BIT;

    VkBufferCreateInfo buffer_info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    // This creates a dummy buffer on vulkan < 1.3 or without maintenance4,
    // in that case size must be > 0 to avoid validation errors.
    buffer_info.size = vk.device_version < VK_API_VERSION_1_3 ? 1 : 0;
    buffer_info.usage = desc.usage;
    if (desc.external) {
        buffer_info.pNext = &external_memory_buffer_info;
    }

    VmaAllocationCreateInfo alloc_create_info = {};
    alloc_create_info.requiredFlags = desc.alloc.memory_properties_required;
    alloc_create_info.preferredFlags = desc.alloc.memory_properties_preferred;
    alloc_create_info.flags = desc.alloc.vma_flags;
    alloc_create_info.usage = desc.alloc.vma_usage;


    // Look for memory type for this allocation
    u32 mem_type_index;
    VkResult vkr = vmaFindMemoryTypeIndexForBufferInfo(vk.vma, &buffer_info, &alloc_create_info, &mem_type_index);
    if (vkr != VK_SUCCESS) {
        return vkr;
    }

    // Create pool that can be exported
    // NOTE: on win32 the cuda samples also specifies a VkExportMemoryWin32HandleInfoKHR for read/write permission and security attributes, but on our system it seems to work without this

    VmaPoolCreateInfo pool_info = {};
    pool_info.memoryTypeIndex = mem_type_index;

    VkExportMemoryAllocateInfo* export_mem_alloc_info = nullptr;
    if (desc.external) {
        // NOTE: The usage of this by vma is delayed, thus the pointer must live as long as the pool.
        // We cannot store this by value in a struct because it would become self-referential and it would not
        // be safely copiable anymore.
        export_mem_alloc_info = new VkExportMemoryAllocateInfo;
        *export_mem_alloc_info = { VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO };
        export_mem_alloc_info->handleTypes = EXTERNAL_MEMORY_HANDLE_TYPE_BIT;
        pool_info.pMemoryAllocateNext = (void*)export_mem_alloc_info;
    }

    VmaPool vma_pool = {};
    vkr = vmaCreatePool(vk.vma, &pool_info, &vma_pool);
    if (vkr != VK_SUCCESS) {
        delete export_mem_alloc_info;
        return vkr;
    }

    *pool = {
        .pool = vma_pool,
        .export_mem_alloc_info = export_mem_alloc_info,
    };

    return VK_SUCCESS;
}

void DestroyPool(Pool* pool, const Context& vk)
{
    vmaDestroyPool(vk.vma, pool->pool);
    delete pool->export_mem_alloc_info;
    *pool = {};
}

#ifdef _WIN32
typedef HANDLE ExternalHandle;
#else
typedef int ExternalHandle;
#endif

VkResult GetExternalHandleForBuffer(ExternalHandle* handle, const Context& vk, const Buffer& buffer) {
#ifdef XPG_MOLTENVK_STATIC
    return VK_ERROR_FEATURE_NOT_PRESENT;
#else
    VmaAllocationInfo alloc_info = {};
    vmaGetAllocationInfo(vk.vma, buffer.allocation, &alloc_info);

#ifdef _WIN32
    VkMemoryGetWin32HandleInfoKHR memory_get_win32_handle_info = { VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR };
    memory_get_win32_handle_info.memory = alloc_info.deviceMemory;
    memory_get_win32_handle_info.handleType = EXTERNAL_MEMORY_HANDLE_TYPE_BIT;

    return vkGetMemoryWin32HandleKHR(vk.device, &memory_get_win32_handle_info, handle);
#else
    VkMemoryGetFdInfoKHR memory_get_fd_info = { VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR };
    memory_get_fd_info.memory = alloc_info.deviceMemory;
    memory_get_fd_info.handleType = EXTERNAL_MEMORY_HANDLE_TYPE_BIT;

    return vkGetMemoryFdKHR(vk.device, &memory_get_fd_info, handle);
#endif
#endif
}

void CloseExternalHandle(ExternalHandle* handle) {
#ifdef _WIN32
    CloseHandle(*handle);
    *handle = INVALID_HANDLE_VALUE;
#else
    close(*handle);
    *handle = -1;
#endif
}

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
    multisample_state.rasterizationSamples = desc.samples;

    VkPipelineDepthStencilStateCreateInfo depth_stencil_state = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    depth_stencil_state.depthTestEnable = desc.depth.test;
    depth_stencil_state.depthWriteEnable = desc.depth.write;
    depth_stencil_state.depthCompareOp = desc.depth.op;

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

    VkDynamicState dynamic_states[3] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_LINE_WIDTH,
    };

    VkPipelineDynamicStateCreateInfo dynamic_state = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynamic_state.dynamicStateCount = ArrayCount(dynamic_states) - (desc.rasterization.dynamic_line_width ? 0 : 1);
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
CreateComputePipeline(ComputePipeline* compute_pipeline, const Context& vk, const ComputePipelineDesc&& desc) {
    VkResult vkr;

    VkPipelineLayoutCreateInfo layout_info = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    layout_info.setLayoutCount = (u32)desc.descriptor_sets.length;
    layout_info.pSetLayouts = desc.descriptor_sets.data;
    layout_info.pushConstantRangeCount = (u32)desc.push_constants.length;
    layout_info.pPushConstantRanges = (VkPushConstantRange*)desc.push_constants.data;

    VkPipelineLayout layout = 0;
    vkr = vkCreatePipelineLayout(vk.device, &layout_info, 0, &layout);
    if (vkr != VK_SUCCESS) {
        return vkr;
    }

    VkComputePipelineCreateInfo pipeline_create_info = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    pipeline_create_info.layout = layout;
    pipeline_create_info.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    pipeline_create_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT ;
    pipeline_create_info.stage.pName = desc.entry;
    pipeline_create_info.stage.module = desc.shader.shader;

    VkPipeline pipeline = 0;
    vkr = vkCreateComputePipelines(vk.device, VK_NULL_HANDLE, 1, &pipeline_create_info, 0, &pipeline);
    if (vkr != VK_SUCCESS) {
        return vkr;
    }

    compute_pipeline->pipeline = pipeline;
    compute_pipeline->layout = layout;

    return VK_SUCCESS;
}

void
DestroyComputePipeline(ComputePipeline* pipeline, const Context& vk) {
    vkDestroyPipelineLayout(vk.device, pipeline->layout, 0);
    vkDestroyPipeline(vk.device, pipeline->pipeline, 0);
    *pipeline = {};
}

static bool
IsDepthFormat(VkFormat format) {
    return format == VK_FORMAT_D16_UNORM ||
           format == VK_FORMAT_X8_D24_UNORM_PACK32 ||
           format == VK_FORMAT_D32_SFLOAT ||
           format == VK_FORMAT_D16_UNORM_S8_UINT ||
           format == VK_FORMAT_D24_UNORM_S8_UINT ||
           format == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

VkResult
CreateImage(Image* image, const Context& vk, const ImageDesc&& desc) {
    // Create a depth buffer.
    VkImageCreateInfo image_create_info = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    image_create_info.imageType = VK_IMAGE_TYPE_2D;
    image_create_info.extent.width = desc.width;
    image_create_info.extent.height = desc.height;
    image_create_info.extent.depth = 1;
    image_create_info.mipLevels = 1;
    image_create_info.arrayLayers = 1;
    image_create_info.format = desc.format;
    image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_create_info.usage = desc.usage;
    image_create_info.samples = desc.samples;

    VmaAllocationCreateInfo alloc_create_info = {};
    alloc_create_info.usage = desc.alloc.vma_usage;
    alloc_create_info.flags = desc.alloc.vma_flags;
    alloc_create_info.requiredFlags = desc.alloc.memory_properties_required;
    alloc_create_info.preferredFlags = desc.alloc.memory_properties_preferred;
    alloc_create_info.priority = 1.0f;

    VkImage img;
    VmaAllocation allocation;
    VkResult vkr = vmaCreateImage(vk.vma, &image_create_info, &alloc_create_info, &img, &allocation, nullptr);
    if (vkr != VK_SUCCESS) {
        return vkr;
    }

    VkImageViewCreateInfo image_view_info = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    image_view_info.image = img;
    image_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    image_view_info.format = desc.format;
    image_view_info.subresourceRange.aspectMask = IsDepthFormat(desc.format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    image_view_info.subresourceRange.levelCount = 1;
    image_view_info.subresourceRange.layerCount = 1;

    VkImageView image_view = 0;
    vkr = vkCreateImageView(vk.device, &image_view_info, 0, &image_view);
    if (vkr != VK_SUCCESS) {
        return vkr;
    }

    image->image = img;
    image->view = image_view;
    image->allocation = allocation;

    return vkr;
}

VkResult
UploadImage(const Image& image, const Context& vk, ArrayView<u8> data, const ImageUploadDesc&& desc)
{
    FormatInfo info = GetFormatInfo(desc.format);
    usize pitch = (usize)desc.width * info.size;
    usize rows = (usize)desc.height;
    if(info.size_of_block_in_bytes > 0) {
        usize blocks_per_row = DivCeil(desc.width, info.block_side_in_pixels);
        pitch = blocks_per_row * info.size_of_block_in_bytes;
        rows = DivCeil(desc.height, info.block_side_in_pixels);
    }
    assert(pitch * rows == data.length);

    VkResult vkr;

    // Create and populate staging buffer
    Buffer staging_buffer = {};
    vkr = CreateBufferFromData(&staging_buffer, vk, data, {
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .alloc = AllocPresets::HostWriteCombining,
    });
    if (vkr != VK_SUCCESS) {
        return vkr;
    }

    // Issue upload
    gfx::BeginCommands(vk.sync_command_pool, vk.sync_command_buffer, vk);

    if (desc.current_image_layout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        gfx::CmdImageBarrier(vk.sync_command_buffer, {
            .src_stage = VK_PIPELINE_STAGE_2_NONE,
            .src_access = 0,
            .dst_stage = VK_PIPELINE_STAGE_2_COPY_BIT,
            .dst_access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .old_layout = desc.current_image_layout,
            .new_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .image = image.image,
        });
    }

    VkBufferImageCopy copy = {};
    copy.bufferOffset = 0;
    // copy.bufferRowLength = pitch; // NOTE: wrong for block compressed format, in that case just use size / height
    // copy.bufferImageHeight = desc.height;
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.mipLevel = 0;
    copy.imageSubresource.baseArrayLayer = 0;
    copy.imageSubresource.layerCount = 1;
    copy.imageExtent.width = desc.width;
    copy.imageExtent.height = desc.height;
    copy.imageExtent.depth = 1;

    vkCmdCopyBufferToImage(vk.sync_command_buffer, staging_buffer.buffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

    if (desc.final_image_layout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        gfx::CmdImageBarrier(vk.sync_command_buffer, {
            .src_stage = VK_PIPELINE_STAGE_2_COPY_BIT,
            .src_access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dst_stage = VK_PIPELINE_STAGE_2_NONE,
            .dst_access = 0,
            .old_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .new_layout = desc.final_image_layout,
            .image = image.image,
        });
    }

    gfx::EndCommands(vk.sync_command_buffer);

    gfx::SubmitSync(vk);

    gfx::DestroyBuffer(&staging_buffer, vk);

    return VK_SUCCESS;
}

VkResult
CreateAndUploadImage(Image* image, const Context& vk, ArrayView<u8> data, VkImageLayout layout, const ImageDesc&& desc)
{
    assert(desc.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    VkResult vkr;
    vkr = CreateImage(image, vk, move(desc));
    if (vkr != VK_SUCCESS) {
        return vkr;
    }
    vkr = UploadImage(*image, vk, data, {
        .width = desc.width,
        .height = desc.height,
        .format = desc.format,
        .final_image_layout = layout,
    });
    if (vkr != VK_SUCCESS) {
        return vkr;
    }
    return VK_SUCCESS;
}

void
DestroyImage(Image* image, const Context& vk)
{
    vkDestroyImageView(vk.device, image->view, 0);
    vmaDestroyImage(vk.vma, image->image, image->allocation);
    *image = {};
}

VkResult CreateDescriptorSet(DescriptorSet* set, const Context& vk, const DescriptorSetDesc&& desc)
{
    VkResult vkr;

    usize N = desc.entries.length;
    Array<VkDescriptorSetLayoutBinding> bindings(N);
    Array<VkDescriptorBindingFlags> flags(N);
    Array<VkDescriptorPoolSize> descriptor_pool_sizes(N);

    for (uint32_t i = 0; i < N; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = desc.entries[i].type;
        bindings[i].descriptorCount = desc.entries[i].count;
        bindings[i].stageFlags = VK_SHADER_STAGE_ALL;
        flags[i] = desc.flags;
        descriptor_pool_sizes[i] = { desc.entries[i].type, desc.entries[i].count };
    }

    VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
    binding_flags.bindingCount = (uint32_t)bindings.length;
    binding_flags.pBindingFlags = flags.data;

    VkDescriptorSetLayoutCreateInfo create_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    if (desc.flags) {
        create_info.pNext = &binding_flags;
    }
    create_info.bindingCount = (uint32_t)bindings.length;
    create_info.pBindings = bindings.data;
    if (desc.flags & VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT) {
        create_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    }

    // Create layout
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    vkr = vkCreateDescriptorSetLayout(vk.device, &create_info, 0, &descriptor_layout);
    if (vkr != VK_SUCCESS) {
        return vkr;
    }

    // Create pool
    VkDescriptorPoolCreateInfo descriptor_pool_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    if (desc.flags & VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT) {
        descriptor_pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    }
    descriptor_pool_info.maxSets = 1;
    descriptor_pool_info.pPoolSizes = descriptor_pool_sizes.data;
    descriptor_pool_info.poolSizeCount = (uint32_t)descriptor_pool_sizes.length;

    VkDescriptorPool descriptor_pool = 0;
    vkr = vkCreateDescriptorPool(vk.device, &descriptor_pool_info, 0, &descriptor_pool);
    if (vkr != VK_SUCCESS) {
        return vkr;
    }

    // Create descriptor set
    VkDescriptorSetAllocateInfo allocate_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocate_info.descriptorPool = descriptor_pool;
    allocate_info.pSetLayouts = &descriptor_layout;
    allocate_info.descriptorSetCount = 1;

    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    vkr = vkAllocateDescriptorSets(vk.device, &allocate_info, &descriptor_set);
    if (vkr != VK_SUCCESS) {
        return vkr;
    }

    set->set = descriptor_set;
    set->layout = descriptor_layout;
    set->pool = descriptor_pool;

    return VK_SUCCESS;
}

void
DestroyDescriptorSet(DescriptorSet* set, const Context& vk)
{
    vkDestroyDescriptorPool(vk.device, set->pool, 0);
    vkDestroyDescriptorSetLayout(vk.device, set->layout, 0);
    *set = {};
}

VkResult
CreateSampler(Sampler* sampler, const Context& vk, const SamplerDesc&& desc) {
    VkSamplerCreateInfo info = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    info.minFilter = desc.min_filter;
    info.magFilter = desc.mag_filter;
    info.mipmapMode = desc.mipmap_mode;
    info.addressModeU = desc.u;
    info.addressModeV = desc.v;
    info.addressModeW = desc.w;
    info.mipLodBias = desc.mip_lod_bias;
    info.anisotropyEnable = desc.anisotroy_enabled;
    info.maxAnisotropy = desc.max_anisotropy;
    info.compareEnable = desc.compare_enable;
    info.compareOp = desc.compare_op;
    info.minLod = desc.min_lod;
    info.maxLod = desc.max_lod;
    info.borderColor = {};
    info.unnormalizedCoordinates = 0;

    VkSampler s = {};
    VkResult vkr = vkCreateSampler(vk.device, &info, 0, &s);
    if (vkr != VK_SUCCESS) {
        return vkr;
    }

    sampler->sampler = s;
    return VK_SUCCESS;
}

void
DestroySampler(Sampler* sampler, const Context& vk) {
    vkDestroySampler(vk.device, sampler->sampler, 0);
    *sampler = {};
}

void
WriteBufferDescriptor(VkDescriptorSet set, const Context& vk, const BufferDescriptorWriteDesc&& write)
{
    assert(
        write.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
        write.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
        write.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC ||
        write.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
    );

    // Prepare descriptor and handle
    VkDescriptorBufferInfo desc_info = {};
    desc_info.buffer = write.buffer;
    desc_info.offset = write.offset;
    desc_info.range = write.size;

    VkWriteDescriptorSet write_descriptor_set = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write_descriptor_set.dstSet = set;
    write_descriptor_set.dstArrayElement = write.element;
    write_descriptor_set.descriptorCount = 1;
    write_descriptor_set.pBufferInfo = &desc_info;
    write_descriptor_set.dstBinding = write.binding;
    write_descriptor_set.descriptorType = write.type;

    // Actually write the descriptor to the GPU visible heap
    vkUpdateDescriptorSets(vk.device, 1, &write_descriptor_set, 0, nullptr);
}

void
WriteImageDescriptor(VkDescriptorSet set, const Context& vk, const ImageDescriptorWriteDesc&& write)
{
    assert(write.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || write.type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);

    // Prepare descriptor and handle
    VkDescriptorImageInfo desc_info = {};
    desc_info.imageView = write.view;
    desc_info.imageLayout = write.layout;

    VkWriteDescriptorSet write_descriptor_set = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write_descriptor_set.dstSet = set;
    write_descriptor_set.dstArrayElement = write.element;
    write_descriptor_set.descriptorCount = 1;
    write_descriptor_set.pImageInfo = &desc_info;
    write_descriptor_set.dstBinding = write.binding;
    write_descriptor_set.descriptorType = write.type;

    // Actually write the descriptor to the GPU visible heap
    vkUpdateDescriptorSets(vk.device, 1, &write_descriptor_set, 0, nullptr);
}

void
WriteSamplerDescriptor(VkDescriptorSet set, const Context& vk, const SamplerDescriptorWriteDesc&& write)
{
    // Prepare descriptor and handle
    VkDescriptorImageInfo desc_info = {};
    desc_info.sampler = write.sampler;

    VkWriteDescriptorSet write_descriptor_set = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write_descriptor_set.dstSet = set;
    write_descriptor_set.dstArrayElement = write.element;
    write_descriptor_set.descriptorCount = 1;
    write_descriptor_set.pImageInfo = &desc_info;
    write_descriptor_set.dstBinding = write.binding;
    write_descriptor_set.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;

    // Actually write the descriptor to the GPU visible heap
    vkUpdateDescriptorSets(vk.device, 1, &write_descriptor_set, 0, nullptr);
}

void
WriteAccelerationStructureDescriptor(VkDescriptorSet set, const Context& vk, const AccelerationStructureDescriptorWriteDesc&& write)
{
    // Prepare descriptor and handle
    VkWriteDescriptorSetAccelerationStructureKHR  write_as = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
    write_as.accelerationStructureCount = 1;
    write_as.pAccelerationStructures = &write.acceleration_structure;

    VkWriteDescriptorSet write_descriptor_set = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write_descriptor_set.pNext = &write_as;
    write_descriptor_set.dstSet = set;
    write_descriptor_set.dstArrayElement = write.element;
    write_descriptor_set.descriptorCount = 1;
    write_descriptor_set.dstBinding = write.binding;
    write_descriptor_set.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    // Actually write the descriptor to the GPU visible heap
    vkUpdateDescriptorSets(vk.device, 1, &write_descriptor_set, 0, nullptr);
}

VkDeviceAddress GetBufferAddress(VkBuffer buffer, VkDevice device)
{
	VkBufferDeviceAddressInfoKHR info = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
	info.buffer = buffer;

	VkDeviceAddress address = vkGetBufferDeviceAddressKHR(device, &info);
	return address;
}

VkResult CreateAccelerationStructure(AccelerationStructure *as, const Context &vk, const AccelerationStructureDesc &&desc)
{
#ifdef XPG_MOLTENVK_STATIC
    return VK_ERROR_FEATURE_NOT_PRESENT;
#else
    VkResult vkr = VK_SUCCESS;

    // Create blases
    usize num_meshes = desc.meshes.length;
	Array<VkAccelerationStructureGeometryKHR> geometries(num_meshes);
	Array<VkAccelerationStructureBuildGeometryInfoKHR> blas_build_infos(num_meshes);

	const usize ALIGNMENT = 256;
	const usize DEFAULT_SCRATCH_SIZE = 32 * 1024 * 1024;

	usize total_as_size = 0;
	usize max_scratch_size = 0;

	Array<usize> acceleration_offsets(num_meshes);
	Array<usize> acceleration_sizes(num_meshes);
	Array<usize> scratch_sizes(num_meshes);

    VkBuildAccelerationStructureFlagsKHR build_flags = desc.prefer_fast_build ? VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR : VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;

    for(usize i = 0; i < num_meshes; i++) {
        // Skip empty meshes
        if(desc.meshes[i].vertices_count == 0 || desc.meshes[i].primitive_count == 0) continue;

        geometries[i] = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
        geometries[i].geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR ;
        geometries[i].flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        geometries[i].geometry.triangles = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
        geometries[i].geometry.triangles.vertexFormat = desc.meshes[i].vertices_format;
        geometries[i].geometry.triangles.vertexData.deviceAddress = desc.meshes[i].vertices_address;
        geometries[i].geometry.triangles.vertexStride = desc.meshes[i].vertices_stride;
        geometries[i].geometry.triangles.maxVertex = desc.meshes[i].vertices_count - 1;
        geometries[i].geometry.triangles.indexType = desc.meshes[i].indices_type;
        geometries[i].geometry.triangles.indexData.deviceAddress = desc.meshes[i].indices_address;

        blas_build_infos[i] = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
        blas_build_infos[i].type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        blas_build_infos[i].flags = build_flags | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
        blas_build_infos[i].mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        blas_build_infos[i].geometryCount = 1;
        blas_build_infos[i].pGeometries = &geometries[i];

        VkAccelerationStructureBuildSizesInfoKHR as_build_sizes = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
        vkGetAccelerationStructureBuildSizesKHR(vk.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &blas_build_infos[i], &desc.meshes[i].primitive_count, &as_build_sizes);

        acceleration_offsets[i] = total_as_size;
        acceleration_sizes[i] = as_build_sizes.accelerationStructureSize;
        scratch_sizes[i] = as_build_sizes.buildScratchSize;

        total_as_size = AlignUp(total_as_size + as_build_sizes.accelerationStructureSize, ALIGNMENT);
        max_scratch_size = Max<u64>(max_scratch_size, as_build_sizes.buildScratchSize);
    }

    gfx::Buffer blas_buffer = {};
    vkr = gfx::CreateBuffer(&blas_buffer, vk, total_as_size, {
        .usage =  VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        .alloc = AllocPresets::Device,
    });
    if(vkr != VK_SUCCESS) return vkr;

    usize scratch_buffer_size = Max(DEFAULT_SCRATCH_SIZE, max_scratch_size);
    gfx::Buffer blas_scratch = {};
    vkr = gfx::CreateBuffer(&blas_scratch, vk, scratch_buffer_size, {
        .usage =  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        .alloc = AllocPresets::Device,
    });

    VkDeviceAddress scratch_address = GetBufferAddress(blas_scratch.buffer, vk.device);

    Array<VkAccelerationStructureKHR> blas(num_meshes);
    Array<VkAccelerationStructureBuildRangeInfoKHR> blas_build_ranges(num_meshes);
    Array<VkAccelerationStructureBuildRangeInfoKHR*> blas_build_range_pointers(num_meshes);

    for(usize i = 0; i < num_meshes; i++) {
		VkAccelerationStructureCreateInfoKHR acceleration_info = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
		acceleration_info.buffer = blas_buffer.buffer;
		acceleration_info.offset = acceleration_offsets[i];
		acceleration_info.size = acceleration_sizes[i];
		acceleration_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

		vkr = vkCreateAccelerationStructureKHR(vk.device, &acceleration_info, nullptr, &blas[i]);
        if(vkr != VK_SUCCESS) return vkr;
    }

    // Build blases
    VkQueryPoolCreateInfo query_create_info = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
	query_create_info.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
	query_create_info.queryCount = num_meshes;

	VkQueryPool query = 0;
	vkr = vkCreateQueryPool(vk.device, &query_create_info, 0, &query);
    if(vkr != VK_SUCCESS) return vkr;


    vkr = gfx::BeginCommands(vk.sync_command_pool, vk.sync_command_buffer, vk);
    if(vkr != VK_SUCCESS) return vkr;

    for(usize start = 0; start < num_meshes;) {
        usize scratch_offset = 0;

        usize i = start;
        while(i < num_meshes && scratch_offset + scratch_sizes[i] <= scratch_buffer_size) {
            blas_build_infos[i].scratchData.deviceAddress = scratch_address + scratch_offset;
            blas_build_infos[i].dstAccelerationStructure = blas[i];
			blas_build_ranges[i].primitiveCount = desc.meshes[i].primitive_count;
            blas_build_range_pointers[i] = &blas_build_ranges[i];

            scratch_offset += scratch_sizes[i];
            i += 1;
        }
        assert(i > start);

        vkCmdBuildAccelerationStructuresKHR(vk.sync_command_buffer, (u32)(i - start), &blas_build_infos[start], &blas_build_range_pointers[start]);
        start = i;

        gfx::CmdMemoryBarrier(vk.sync_command_buffer, {
            .src_stage = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            .src_access = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            .dst_stage = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            .dst_access = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        });
    }

    vkCmdResetQueryPool(vk.sync_command_buffer, query, 0, num_meshes);
    vkCmdWriteAccelerationStructuresPropertiesKHR(vk.sync_command_buffer, blas.length, blas.data, VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, query, 0);
    vkr = gfx::EndCommands(vk.sync_command_buffer);
    if(vkr != VK_SUCCESS) return vkr;

    vkr = gfx::SubmitSync(vk);
    if(vkr != VK_SUCCESS) return vkr;

    Array<VkDeviceSize> compacted_sizes(num_meshes);
    vkr = vkGetQueryPoolResults(vk.device, query,  0, blas.length, compacted_sizes.size_in_bytes(), compacted_sizes.data, sizeof(VkDeviceSize), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if(vkr != VK_SUCCESS) return vkr;

    vkDestroyQueryPool(vk.device, query, 0);

    gfx::DestroyBuffer(&blas_scratch, vk);

    // Compact blases
    Array<usize> compacted_offsets(num_meshes);
    usize total_compacted_size = 0;
    for(usize i = 0; i < num_meshes; i++) {
        compacted_offsets[i] += total_compacted_size;
        total_compacted_size = AlignUp(total_compacted_size + compacted_sizes[i], ALIGNMENT);
    }

    gfx::Buffer compacted_blas_buffer = {};
    vkr = gfx::CreateBuffer(&compacted_blas_buffer, vk, total_compacted_size, {
        .usage =  VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        .alloc = AllocPresets::Device,
    });
    if(vkr != VK_SUCCESS) return vkr;

    Array<VkAccelerationStructureKHR> compacted_blas(num_meshes);
    for(usize i = 0; i < num_meshes; i++) {
		VkAccelerationStructureCreateInfoKHR acceleration_info = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
		acceleration_info.buffer = compacted_blas_buffer.buffer;
		acceleration_info.offset = compacted_offsets[i];
		acceleration_info.size = compacted_sizes[i];
		acceleration_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		vkr = vkCreateAccelerationStructureKHR(vk.device, &acceleration_info, nullptr, &compacted_blas[i]);
        if(vkr != VK_SUCCESS) return vkr;
    }

    vkr = gfx::BeginCommands(vk.sync_command_pool, vk.sync_command_buffer, vk);
    if(vkr != VK_SUCCESS) return vkr;

    for(usize i = 0; i < num_meshes; i++) {
		VkCopyAccelerationStructureInfoKHR copy_info = { VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR };
		copy_info.src = blas[i];
		copy_info.dst = compacted_blas[i];
		copy_info.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
		vkCmdCopyAccelerationStructureKHR(vk.sync_command_buffer, &copy_info);
    }

    vkr = gfx::EndCommands(vk.sync_command_buffer);
    if(vkr != VK_SUCCESS) return vkr;

    vkr = gfx::SubmitSync(vk);
    if(vkr != VK_SUCCESS) return vkr;

    for(usize i = 0; i < num_meshes; i++) {
        vkDestroyAccelerationStructureKHR(vk.device, blas[i], 0);
    }
    blas.clear();

    gfx::DestroyBuffer(&blas_buffer, vk);

    // Populate instances
    gfx::Buffer instances_buffer;
    gfx::CreateBuffer(&instances_buffer, vk, sizeof(VkAccelerationStructureInstanceKHR) * num_meshes, {
        .usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        .alloc = AllocPresets::DeviceMapped,
    });

    for(usize i = 0; i < num_meshes; i++) {
        VkAccelerationStructureDeviceAddressInfoKHR info = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
        info.accelerationStructure = compacted_blas[i];
        VkDeviceAddress address = vkGetAccelerationStructureDeviceAddressKHR(vk.device, &info);

        VkAccelerationStructureInstanceKHR instance = {};
        glm::mat3x4 transform = transpose(desc.meshes[i].transform);
        memcpy(instance.transform.matrix[0], &transform[0][0], sizeof(float) * 4);
        memcpy(instance.transform.matrix[1], &transform[1][0], sizeof(float) * 4);
        memcpy(instance.transform.matrix[2], &transform[2][0], sizeof(float) * 4);
        instance.instanceCustomIndex = i;
        instance.mask = 0xFF;
        instance.flags = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
        instance.accelerationStructureReference = address;

        memcpy(instances_buffer.map.data + i * sizeof(VkAccelerationStructureInstanceKHR), &instance, sizeof(VkAccelerationStructureInstanceKHR));
    }

    // Create tlas
    VkAccelerationStructureGeometryKHR tlas_geometry = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
	tlas_geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
	tlas_geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
	tlas_geometry.geometry.instances.data.deviceAddress = GetBufferAddress(instances_buffer.buffer, vk.device);

	VkAccelerationStructureBuildGeometryInfoKHR tlas_build_info = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
	tlas_build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	tlas_build_info.flags = build_flags;
	tlas_build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	tlas_build_info.geometryCount = 1;
	tlas_build_info.pGeometries = &tlas_geometry;

	VkAccelerationStructureBuildSizesInfoKHR tlas_build_sizes = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
    u32 max_count = num_meshes;
	vkGetAccelerationStructureBuildSizesKHR(vk.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlas_build_info, &max_count, &tlas_build_sizes);

    gfx::Buffer tlas_buffer = {};
	vkr = gfx::CreateBuffer(&tlas_buffer, vk, tlas_build_sizes.accelerationStructureSize, {
        .usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
        .alloc = AllocPresets::Device,
    });
    if(vkr != VK_SUCCESS) return vkr;

    gfx::Buffer tlas_scratch_buffer = {};
	vkr = gfx::CreateBuffer(&tlas_scratch_buffer, vk, tlas_build_sizes.buildScratchSize, {
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        .alloc = AllocPresets::Device,
    });
    if(vkr != VK_SUCCESS) return vkr;

	VkAccelerationStructureCreateInfoKHR tlas_info = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
	tlas_info.buffer = tlas_buffer.buffer;
	tlas_info.size = tlas_build_sizes.accelerationStructureSize;
	tlas_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

	VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
	vkr = vkCreateAccelerationStructureKHR(vk.device, &tlas_info, nullptr, &tlas);
    if(vkr != VK_SUCCESS) return vkr;

    // Build tlas
	tlas_build_info.srcAccelerationStructure = tlas;
	tlas_build_info.dstAccelerationStructure = tlas;
	tlas_build_info.scratchData.deviceAddress = GetBufferAddress(tlas_scratch_buffer.buffer, vk.device);

	VkAccelerationStructureBuildRangeInfoKHR build_range = {};
	build_range.primitiveCount = num_meshes;

	const VkAccelerationStructureBuildRangeInfoKHR* build_range_ptr = &build_range;

    vkr = gfx::BeginCommands(vk.sync_command_pool, vk.sync_command_buffer, vk);
    if(vkr != VK_SUCCESS) return vkr;

	vkCmdBuildAccelerationStructuresKHR(vk.sync_command_buffer, 1, &tlas_build_info, &build_range_ptr);

    vkr = gfx::EndCommands(vk.sync_command_buffer);
    if(vkr != VK_SUCCESS) return vkr;
    vkr = gfx::SubmitSync(vk);
    if(vkr != VK_SUCCESS) return vkr;

    gfx::DestroyBuffer(&tlas_scratch_buffer, vk);

    as->blas = std::move(compacted_blas);
    as->blas_buffer = compacted_blas_buffer;
    as->tlas = tlas;
    as->tlas_buffer = tlas_buffer;
    as->instances_buffer = instances_buffer;

    return VK_SUCCESS;
#endif
}

void DestroyAccelerationStructure(AccelerationStructure* as, const Context& vk) {
#ifndef XPG_MOLTENVK_STATIC
    vkDestroyAccelerationStructureKHR(vk.device, as->tlas, 0);
    as->tlas = VK_NULL_HANDLE;

    for(usize i = 0; i < as->blas.length; i++) {
        vkDestroyAccelerationStructureKHR(vk.device, as->blas[i], 0);
    }
    as->blas.clear();

    gfx::DestroyBuffer(&as->tlas_buffer, vk);
    gfx::DestroyBuffer(&as->instances_buffer, vk);
    gfx::DestroyBuffer(&as->blas_buffer, vk);
#endif
}

// TODO: Fix missing block formats info
FormatInfo GetFormatInfo(VkFormat format)
{
    switch(format) {
        case VK_FORMAT_UNDEFINED:                                   return { 0, 0, 0, 0 };
        case VK_FORMAT_R4G4_UNORM_PACK8:                            return { 1, 2, 0, 0 };
        case VK_FORMAT_R4G4B4A4_UNORM_PACK16:                       return { 2, 4, 0, 0 };
        case VK_FORMAT_B4G4R4A4_UNORM_PACK16:                       return { 2, 4, 0, 0 };
        case VK_FORMAT_R5G6B5_UNORM_PACK16:                         return { 2, 3, 0, 0 };
        case VK_FORMAT_B5G6R5_UNORM_PACK16:                         return { 2, 3, 0, 0 };
        case VK_FORMAT_R5G5B5A1_UNORM_PACK16:                       return { 2, 4, 0, 0 };
        case VK_FORMAT_B5G5R5A1_UNORM_PACK16:                       return { 2, 4, 0, 0 };
        case VK_FORMAT_A1R5G5B5_UNORM_PACK16:                       return { 2, 4, 0, 0 };
        case VK_FORMAT_R8_UNORM:                                    return { 1, 1, 0, 0 };
        case VK_FORMAT_R8_SNORM:                                    return { 1, 1, 0, 0 };
        case VK_FORMAT_R8_USCALED:                                  return { 1, 1, 0, 0 };
        case VK_FORMAT_R8_SSCALED:                                  return { 1, 1, 0, 0 };
        case VK_FORMAT_R8_UINT:                                     return { 1, 1, 0, 0 };
        case VK_FORMAT_R8_SINT:                                     return { 1, 1, 0, 0 };
        case VK_FORMAT_R8_SRGB:                                     return { 1, 1, 0, 0 };
        case VK_FORMAT_R8G8_UNORM:                                  return { 2, 2, 0, 0 };
        case VK_FORMAT_R8G8_SNORM:                                  return { 2, 2, 0, 0 };
        case VK_FORMAT_R8G8_USCALED:                                return { 2, 2, 0, 0 };
        case VK_FORMAT_R8G8_SSCALED:                                return { 2, 2, 0, 0 };
        case VK_FORMAT_R8G8_UINT:                                   return { 2, 2, 0, 0 };
        case VK_FORMAT_R8G8_SINT:                                   return { 2, 2, 0, 0 };
        case VK_FORMAT_R8G8_SRGB:                                   return { 2, 2, 0, 0 };
        case VK_FORMAT_R8G8B8_UNORM:                                return { 3, 3, 0, 0 };
        case VK_FORMAT_R8G8B8_SNORM:                                return { 3, 3, 0, 0 };
        case VK_FORMAT_R8G8B8_USCALED:                              return { 3, 3, 0, 0 };
        case VK_FORMAT_R8G8B8_SSCALED:                              return { 3, 3, 0, 0 };
        case VK_FORMAT_R8G8B8_UINT:                                 return { 3, 3, 0, 0 };
        case VK_FORMAT_R8G8B8_SINT:                                 return { 3, 3, 0, 0 };
        case VK_FORMAT_R8G8B8_SRGB:                                 return { 3, 3, 0, 0 };
        case VK_FORMAT_B8G8R8_UNORM:                                return { 3, 3, 0, 0 };
        case VK_FORMAT_B8G8R8_SNORM:                                return { 3, 3, 0, 0 };
        case VK_FORMAT_B8G8R8_USCALED:                              return { 3, 3, 0, 0 };
        case VK_FORMAT_B8G8R8_SSCALED:                              return { 3, 3, 0, 0 };
        case VK_FORMAT_B8G8R8_UINT:                                 return { 3, 3, 0, 0 };
        case VK_FORMAT_B8G8R8_SINT:                                 return { 3, 3, 0, 0 };
        case VK_FORMAT_B8G8R8_SRGB:                                 return { 3, 3, 0, 0 };
        case VK_FORMAT_R8G8B8A8_UNORM:                              return { 4, 4, 0, 0 };
        case VK_FORMAT_R8G8B8A8_SNORM:                              return { 4, 4, 0, 0 };
        case VK_FORMAT_R8G8B8A8_USCALED:                            return { 4, 4, 0, 0 };
        case VK_FORMAT_R8G8B8A8_SSCALED:                            return { 4, 4, 0, 0 };
        case VK_FORMAT_R8G8B8A8_UINT:                               return { 4, 4, 0, 0 };
        case VK_FORMAT_R8G8B8A8_SINT:                               return { 4, 4, 0, 0 };
        case VK_FORMAT_R8G8B8A8_SRGB:                               return { 4, 4, 0, 0 };
        case VK_FORMAT_B8G8R8A8_UNORM:                              return { 4, 4, 0, 0 };
        case VK_FORMAT_B8G8R8A8_SNORM:                              return { 4, 4, 0, 0 };
        case VK_FORMAT_B8G8R8A8_USCALED:                            return { 4, 4, 0, 0 };
        case VK_FORMAT_B8G8R8A8_SSCALED:                            return { 4, 4, 0, 0 };
        case VK_FORMAT_B8G8R8A8_UINT:                               return { 4, 4, 0, 0 };
        case VK_FORMAT_B8G8R8A8_SINT:                               return { 4, 4, 0, 0 };
        case VK_FORMAT_B8G8R8A8_SRGB:                               return { 4, 4, 0, 0 };
        case VK_FORMAT_A8B8G8R8_UNORM_PACK32:                       return { 4, 4, 0, 0 };
        case VK_FORMAT_A8B8G8R8_SNORM_PACK32:                       return { 4, 4, 0, 0 };
        case VK_FORMAT_A8B8G8R8_USCALED_PACK32:                     return { 4, 4, 0, 0 };
        case VK_FORMAT_A8B8G8R8_SSCALED_PACK32:                     return { 4, 4, 0, 0 };
        case VK_FORMAT_A8B8G8R8_UINT_PACK32:                        return { 4, 4, 0, 0 };
        case VK_FORMAT_A8B8G8R8_SINT_PACK32:                        return { 4, 4, 0, 0 };
        case VK_FORMAT_A8B8G8R8_SRGB_PACK32:                        return { 4, 4, 0, 0 };
        case VK_FORMAT_A2R10G10B10_UNORM_PACK32:                    return { 4, 4, 0, 0 };
        case VK_FORMAT_A2R10G10B10_SNORM_PACK32:                    return { 4, 4, 0, 0 };
        case VK_FORMAT_A2R10G10B10_USCALED_PACK32:                  return { 4, 4, 0, 0 };
        case VK_FORMAT_A2R10G10B10_SSCALED_PACK32:                  return { 4, 4, 0, 0 };
        case VK_FORMAT_A2R10G10B10_UINT_PACK32:                     return { 4, 4, 0, 0 };
        case VK_FORMAT_A2R10G10B10_SINT_PACK32:                     return { 4, 4, 0, 0 };
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:                    return { 4, 4, 0, 0 };
        case VK_FORMAT_A2B10G10R10_SNORM_PACK32:                    return { 4, 4, 0, 0 };
        case VK_FORMAT_A2B10G10R10_USCALED_PACK32:                  return { 4, 4, 0, 0 };
        case VK_FORMAT_A2B10G10R10_SSCALED_PACK32:                  return { 4, 4, 0, 0 };
        case VK_FORMAT_A2B10G10R10_UINT_PACK32:                     return { 4, 4, 0, 0 };
        case VK_FORMAT_A2B10G10R10_SINT_PACK32:                     return { 4, 4, 0, 0 };
        case VK_FORMAT_R16_UNORM:                                   return { 2, 1, 0, 0 };
        case VK_FORMAT_R16_SNORM:                                   return { 2, 1, 0, 0 };
        case VK_FORMAT_R16_USCALED:                                 return { 2, 1, 0, 0 };
        case VK_FORMAT_R16_SSCALED:                                 return { 2, 1, 0, 0 };
        case VK_FORMAT_R16_UINT:                                    return { 2, 1, 0, 0 };
        case VK_FORMAT_R16_SINT:                                    return { 2, 1, 0, 0 };
        case VK_FORMAT_R16_SFLOAT:                                  return { 2, 1, 0, 0 };
        case VK_FORMAT_R16G16_UNORM:                                return { 4, 2, 0, 0 };
        case VK_FORMAT_R16G16_SNORM:                                return { 4, 2, 0, 0 };
        case VK_FORMAT_R16G16_USCALED:                              return { 4, 2, 0, 0 };
        case VK_FORMAT_R16G16_SSCALED:                              return { 4, 2, 0, 0 };
        case VK_FORMAT_R16G16_UINT:                                 return { 4, 2, 0, 0 };
        case VK_FORMAT_R16G16_SINT:                                 return { 4, 2, 0, 0 };
        case VK_FORMAT_R16G16_SFLOAT:                               return { 4, 2, 0, 0 };
        case VK_FORMAT_R16G16B16_UNORM:                             return { 6, 3, 0, 0 };
        case VK_FORMAT_R16G16B16_SNORM:                             return { 6, 3, 0, 0 };
        case VK_FORMAT_R16G16B16_USCALED:                           return { 6, 3, 0, 0 };
        case VK_FORMAT_R16G16B16_SSCALED:                           return { 6, 3, 0, 0 };
        case VK_FORMAT_R16G16B16_UINT:                              return { 6, 3, 0, 0 };
        case VK_FORMAT_R16G16B16_SINT:                              return { 6, 3, 0, 0 };
        case VK_FORMAT_R16G16B16_SFLOAT:                            return { 6, 3, 0, 0 };
        case VK_FORMAT_R16G16B16A16_UNORM:                          return { 8, 4, 0, 0 };
        case VK_FORMAT_R16G16B16A16_SNORM:                          return { 8, 4, 0, 0 };
        case VK_FORMAT_R16G16B16A16_USCALED:                        return { 8, 4, 0, 0 };
        case VK_FORMAT_R16G16B16A16_SSCALED:                        return { 8, 4, 0, 0 };
        case VK_FORMAT_R16G16B16A16_UINT:                           return { 8, 4, 0, 0 };
        case VK_FORMAT_R16G16B16A16_SINT:                           return { 8, 4, 0, 0 };
        case VK_FORMAT_R16G16B16A16_SFLOAT:                         return { 8, 4, 0, 0 };
        case VK_FORMAT_R32_UINT:                                    return { 4, 1, 0, 0 };
        case VK_FORMAT_R32_SINT:                                    return { 4, 1, 0, 0 };
        case VK_FORMAT_R32_SFLOAT:                                  return { 4, 1, 0, 0 };
        case VK_FORMAT_R32G32_UINT:                                 return { 8, 2, 0, 0 };
        case VK_FORMAT_R32G32_SINT:                                 return { 8, 2, 0, 0 };
        case VK_FORMAT_R32G32_SFLOAT:                               return { 8, 2, 0, 0 };
        case VK_FORMAT_R32G32B32_UINT:                              return {12, 3, 0, 0 };
        case VK_FORMAT_R32G32B32_SINT:                              return {12, 3, 0, 0 };
        case VK_FORMAT_R32G32B32_SFLOAT:                            return {12, 3, 0, 0 };
        case VK_FORMAT_R32G32B32A32_UINT:                           return {16, 4, 0, 0 };
        case VK_FORMAT_R32G32B32A32_SINT:                           return {16, 4, 0, 0 };
        case VK_FORMAT_R32G32B32A32_SFLOAT:                         return {16, 4, 0, 0 };
        case VK_FORMAT_R64_UINT:                                    return { 8, 1, 0, 0 };
        case VK_FORMAT_R64_SINT:                                    return { 8, 1, 0, 0 };
        case VK_FORMAT_R64_SFLOAT:                                  return { 8, 1, 0, 0 };
        case VK_FORMAT_R64G64_UINT:                                 return {16, 2, 0, 0 };
        case VK_FORMAT_R64G64_SINT:                                 return {16, 2, 0, 0 };
        case VK_FORMAT_R64G64_SFLOAT:                               return {16, 2, 0, 0 };
        case VK_FORMAT_R64G64B64_UINT:                              return {24, 3, 0, 0 };
        case VK_FORMAT_R64G64B64_SINT:                              return {24, 3, 0, 0 };
        case VK_FORMAT_R64G64B64_SFLOAT:                            return {24, 3, 0, 0 };
        case VK_FORMAT_R64G64B64A64_UINT:                           return {32, 4, 0, 0 };
        case VK_FORMAT_R64G64B64A64_SINT:                           return {32, 4, 0, 0 };
        case VK_FORMAT_R64G64B64A64_SFLOAT:                         return {32, 4, 0, 0 };
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32:                     return { 4, 3, 0, 0 };
        case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:                      return { 4, 3, 0, 0 };
        case VK_FORMAT_D16_UNORM:                                   return { 2, 1, 0, 0 };
        case VK_FORMAT_X8_D24_UNORM_PACK32:                         return { 4, 1, 0, 0 };
        case VK_FORMAT_D32_SFLOAT:                                  return { 4, 1, 0, 0 };
        case VK_FORMAT_S8_UINT:                                     return { 1, 1, 0, 0 };
        case VK_FORMAT_D16_UNORM_S8_UINT:                           return { 3, 2, 0, 0 };
        case VK_FORMAT_D24_UNORM_S8_UINT:                           return { 4, 2, 0, 0 };
        case VK_FORMAT_D32_SFLOAT_S8_UINT:                          return { 8, 2, 0, 0 };
        case VK_FORMAT_BC1_RGB_UNORM_BLOCK:                         return { 8, 4, 0, 0 };
        case VK_FORMAT_BC1_RGB_SRGB_BLOCK:                          return { 8, 4, 0, 0 };
        case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:                        return { 8, 4, 0, 0 };
        case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:                         return { 8, 4, 0, 0 };
        case VK_FORMAT_BC2_UNORM_BLOCK:                             return {16, 4, 0, 0 };
        case VK_FORMAT_BC2_SRGB_BLOCK:                              return {16, 4, 0, 0 };
        case VK_FORMAT_BC3_UNORM_BLOCK:                             return {16, 4, 0, 0 };
        case VK_FORMAT_BC3_SRGB_BLOCK:                              return {16, 4, 0, 0 };
        case VK_FORMAT_BC4_UNORM_BLOCK:                             return { 8, 4, 0, 0 };
        case VK_FORMAT_BC4_SNORM_BLOCK:                             return { 8, 4, 0, 0 };
        case VK_FORMAT_BC5_UNORM_BLOCK:                             return {16, 4, 0, 0 };
        case VK_FORMAT_BC5_SNORM_BLOCK:                             return {16, 4, 0, 0 };
        case VK_FORMAT_BC6H_UFLOAT_BLOCK:                           return {16, 4, 0, 0 };
        case VK_FORMAT_BC6H_SFLOAT_BLOCK:                           return {16, 4, 0, 0 };
        case VK_FORMAT_BC7_UNORM_BLOCK:                             return {16, 4, 16, 4 };
        case VK_FORMAT_BC7_SRGB_BLOCK:                              return {16, 4, 16, 4 };
        case VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK:                     return { 8, 3, 0, 0 };
        case VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK:                      return { 8, 3, 0, 0 };
        case VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK:                   return { 8, 4, 0, 0 };
        case VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK:                    return { 8, 4, 0, 0 };
        case VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK:                   return {16, 4, 0, 0 };
        case VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK:                    return {16, 4, 0, 0 };
        case VK_FORMAT_EAC_R11_UNORM_BLOCK:                         return { 8, 1, 0, 0 };
        case VK_FORMAT_EAC_R11_SNORM_BLOCK:                         return { 8, 1, 0, 0 };
        case VK_FORMAT_EAC_R11G11_UNORM_BLOCK:                      return {16, 2, 0, 0 };
        case VK_FORMAT_EAC_R11G11_SNORM_BLOCK:                      return {16, 2, 0, 0 };
        case VK_FORMAT_ASTC_4x4_UNORM_BLOCK:                        return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_4x4_SRGB_BLOCK:                         return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_5x4_UNORM_BLOCK:                        return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_5x4_SRGB_BLOCK:                         return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_5x5_UNORM_BLOCK:                        return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_5x5_SRGB_BLOCK:                         return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_6x5_UNORM_BLOCK:                        return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_6x5_SRGB_BLOCK:                         return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_6x6_UNORM_BLOCK:                        return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_6x6_SRGB_BLOCK:                         return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_8x5_UNORM_BLOCK:                        return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_8x5_SRGB_BLOCK:                         return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_8x6_UNORM_BLOCK:                        return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_8x6_SRGB_BLOCK:                         return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_8x8_UNORM_BLOCK:                        return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_8x8_SRGB_BLOCK:                         return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_10x5_UNORM_BLOCK:                       return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_10x5_SRGB_BLOCK:                        return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_10x6_UNORM_BLOCK:                       return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_10x6_SRGB_BLOCK:                        return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_10x8_UNORM_BLOCK:                       return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_10x8_SRGB_BLOCK:                        return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_10x10_UNORM_BLOCK:                      return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_10x10_SRGB_BLOCK:                       return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_12x10_UNORM_BLOCK:                      return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_12x10_SRGB_BLOCK:                       return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_12x12_UNORM_BLOCK:                      return {16, 4, 0, 0 };
        case VK_FORMAT_ASTC_12x12_SRGB_BLOCK:                       return {16, 4, 0, 0 };
        case VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG:                 return { 8, 4, 0, 0 };
        case VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG:                 return { 8, 4, 0, 0 };
        case VK_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG:                 return { 8, 4, 0, 0 };
        case VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG:                 return { 8, 4, 0, 0 };
        case VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG:                  return { 8, 4, 0, 0 };
        case VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG:                  return { 8, 4, 0, 0 };
        case VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG:                  return { 8, 4, 0, 0 };
        case VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG:                  return { 8, 4, 0, 0 };
        case VK_FORMAT_R10X6_UNORM_PACK16:                          return { 2, 1, 0, 0 };
        case VK_FORMAT_R10X6G10X6_UNORM_2PACK16:                    return { 4, 2, 0, 0 };
        case VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16:          return { 8, 4, 0, 0 };
        case VK_FORMAT_R12X4_UNORM_PACK16:                          return { 2, 1, 0, 0 };
        case VK_FORMAT_R12X4G12X4_UNORM_2PACK16:                    return { 4, 2, 0, 0 };
        case VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16:          return { 8, 4, 0, 0 };
        case VK_FORMAT_G8B8G8R8_422_UNORM:                          return { 4, 4, 0, 0 };
        case VK_FORMAT_B8G8R8G8_422_UNORM:                          return { 4, 4, 0, 0 };
        case VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16:      return { 8, 4, 0, 0 };
        case VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16:      return { 8, 4, 0, 0 };
        case VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16:      return { 8, 4, 0, 0 };
        case VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16:      return { 8, 4, 0, 0 };
        case VK_FORMAT_G16B16G16R16_422_UNORM:                      return { 8, 4, 0, 0 };
        case VK_FORMAT_B16G16R16G16_422_UNORM:                      return { 8, 4, 0, 0 };
        case VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM:                   return { 6, 3, 0, 0 };
        case VK_FORMAT_G8_B8R8_2PLANE_420_UNORM:                    return { 6, 3, 0, 0 };
        case VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16:  return {12, 3, 0, 0 };
        case VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16:   return {12, 3, 0, 0 };
        case VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16:  return {12, 3, 0, 0 };
        case VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16:   return {12, 3, 0, 0 };
        case VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM:                return {12, 3, 0, 0 };
        case VK_FORMAT_G16_B16R16_2PLANE_420_UNORM:                 return {12, 3, 0, 0 };
        case VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM:                   return { 4, 3, 0, 0 };
        case VK_FORMAT_G8_B8R8_2PLANE_422_UNORM:                    return { 4, 3, 0, 0 };
        case VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16:  return { 8, 3, 0, 0 };
        case VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16:   return { 8, 3, 0, 0 };
        case VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16:  return { 8, 3, 0, 0 };
        case VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16:   return { 8, 3, 0, 0 };
        case VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM:                return { 8, 3, 0, 0 };
        case VK_FORMAT_G16_B16R16_2PLANE_422_UNORM:                 return { 8, 3, 0, 0 };
        case VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM:                   return { 3, 3, 0, 0 };
        case VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16:  return { 6, 3, 0, 0 };
        case VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16:  return { 6, 3, 0, 0 };
        case VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM:                return { 6, 3, 0, 0 };
        default: return {};
    }
}

} // namespace gfx
} // namespace xpg
