#include <stdio.h>
#include <stdlib.h>

#define VOLK_IMPLEMENTATION
#include <volk.h>

#define _GLFW_VULKAN_STATIC
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>


static VkBool32 VKAPI_CALL 
debug_report_callback(VkDebugReportFlagsEXT flags, 
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

int main(int argc, char** argv) {
    glm::vec3 v = glm::vec3(0.0f);

    // Load vulkan.
    volkInitialize();

    // VkInstance instance;
    // volkLoadInstance(instance);

    // Initialize glfw.
    glfwInit();

    // Check if device supports vulkan.
    if (!glfwVulkanSupported()) {
        printf("Vulkan found!\n");
    }

    // Get instance extensions required by GLFW.
    uint32_t count;
    const char** extensions = glfwGetRequiredInstanceExtensions(&count);

    // Create vulkan instance with the required extensions.
    VkInstanceCreateInfo instance_create_info = {};
    instance_create_info.enabledExtensionCount = count;
    instance_create_info.ppEnabledExtensionNames = extensions;

#if 0
    // Check if queue family supports image presentation.
    if (!glfwGetPhysicalDevicePresentationSupport(instance, physical_device, queue_family_index)) {

    }
#endif

    // Create window.
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(640, 480, "XGP", NULL, NULL);

#if 0
    // Create window surface.
    VkSurfaceKHR surface;
    VkResult err = glfwCreateWindowSurface(instance, window, NULL, &surface);
    if (err) {
    }
#endif

    system("pause");
}