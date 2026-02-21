// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

#pragma once

// External
#ifndef XPG_MOLTENVK_STATIC
#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <volk.h>                           // Vulkan
#else
// If linking statically with MoltenVK we use the standard
// headers directly because we don't dynamically load with volk.
#include <vulkan/vulkan.h>
#endif

#define VMA_STATIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>                   // Vulkan Memory Allocator

#define _GLFW_VULKAN_STATIC
#include <GLFW/glfw3.h>                     // GLFW Windowing
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif
#undef APIENTRY

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>                      // GLM math
#include <glm/gtc/matrix_transform.hpp>     // GLM matrix ops

// Internal
#include "function.h"
#include "array.h"

namespace xpg {
namespace gfx {

// Missing helpers:
// [ ] commands
//      [x] Dynamic rendering
//      [ ] Dispatches
//      [ ] Copies
// [ ] command buffer / pools creation (duplicate for sync and per frame)
// [ ] descriptors:
//     [x] normal set creation
//     [x] writes
//     [ ] bindless descriptors management helpers

const uint32_t ANY_POSITION = GLFW_ANY_POSITION;

struct Window;

enum class Result
{
    SUCCESS,

    GLFW_INITIALIZATION_FAILED,
    API_ERROR,
    API_OUT_OF_MEMORY,
    INVALID_VERSION,
    LAYER_NOT_PRESENT,
    EXTENSION_NOT_PRESENT,
    NO_VALID_DEVICE_FOUND,
    DEVICE_CREATION_FAILED,
    SWAPCHAIN_CREATION_FAILED,
    SWAPCHAIN_OUT_OF_DATE,
    SURFACE_CREATION_FAILED,
    VMA_CREATION_FAILED,
    WINDOW_CREATION_FAILED,
};

struct DeviceFeatures {
    enum Flags: u64 {
        NONE                                    =         0,
        DYNAMIC_RENDERING                       = 1ull << 0,
        SYNCHRONIZATION_2                       = 1ull << 1,
        DESCRIPTOR_INDEXING                     = 1ull << 2,
        SCALAR_BLOCK_LAYOUT                     = 1ull << 3,
        RAY_QUERY                               = 1ull << 4,
        RAY_TRACING_PIPELINE                    = 1ull << 5,
        EXTERNAL_RESOURCES                      = 1ull << 6,
        HOST_QUERY_RESET                        = 1ull << 7,
        CALIBRATED_TIMESTAMPS                   = 1ull << 8,
        TIMELINE_SEMAPHORES                     = 1ull << 9,
        WIDE_LINES                              = 1ull << 10,
        SHADER_DRAW_PARAMETERS                  = 1ull << 11,
        STORAGE_IMAGE_READ_WRITE_WITHOUT_FORMAT = 1ull << 12,
        SHADER_FLOAT16_INT8                     = 1ull << 13,
        SHADER_INT16                            = 1ull << 14,
        SHADER_INT64                            = 1ull << 15,
        SHADER_SUBGROUP_EXTENDED_TYPES          = 1ull << 16,
        STORAGE_8BIT                            = 1ull << 17,
        STORAGE_16BIT                           = 1ull << 18,
        DRAW_INDIRECT_COUNT                     = 1ull << 19,
        MESH_SHADER                             = 1ull << 20,
        FRAGMENT_SHADER_BARYCENTRIC             = 1ull << 21,
        SUBGROUP_SIZE_CONTROL                   = 1ull << 22,
    };

    DeviceFeatures() {};
    DeviceFeatures(Flags flags): flags(flags) {}
    DeviceFeatures(uint64_t flags): flags((Flags)flags) {}

    operator bool() {
        return (uint64_t)flags != 0;
    }

    DeviceFeatures operator|(const Flags& b) const {
        return (Flags)((uint64_t)flags | (uint64_t)b);
    }

    DeviceFeatures operator|(const DeviceFeatures& b) const {
        return (Flags)((uint64_t)flags | (uint64_t)b.flags);
    }

    DeviceFeatures operator&(const Flags& b) const {
        return (Flags)((uint64_t)flags & (uint64_t)b);
    }

    DeviceFeatures operator&(const DeviceFeatures& b) const {
        return (Flags)((uint64_t)flags & (uint64_t)b.flags);
    }

    DeviceFeatures operator~() const {
        return (Flags)(~(uint64_t)flags);
    }

    bool operator==(const DeviceFeatures& b) const {
        return ((uint64_t)flags == (uint64_t)b.flags);
    }

    bool operator!=(const DeviceFeatures& b) const {
        return ((uint64_t)flags != (uint64_t)b.flags);
    }

    Flags flags = Flags::NONE;
};

//- Instance
struct Instance
{
    u32 version;
    VkInstance instance;

    // Debug
    bool debug_utils_enabled;
    VkDebugUtilsMessengerEXT debug_messenger;
};

struct InstanceDesc
{
    u32 minimum_api_version = VK_API_VERSION_1_1;

    // Add required instance extensions to enable presentation
    bool require_presentation = true;

    // Debug utils, adds the instance extension VK_EXT_debug_utils to enable
    // debug report, names and markers.
    VkBool32 enable_debug_utils = false;

    // Enable validation layer. If this is set to true enable_debug_utils
    // is also implicitly enabled to report validation errors.
    VkBool32 enable_validation_layer = false;

    // Validation features, they require the validation to be enabled
    VkBool32 enable_gpu_based_validation = false;
    VkBool32 enable_synchronization_validation = false;
    VkBool32 enable_shader_debug_printf = false;

    uint32_t shader_debug_printf_buffer_size = 4096;
};

struct Device
{
    u32 version;
    VkPhysicalDevice physical_device;
    VkDevice device;

    VmaAllocator vma;

    // Device features
    DeviceFeatures features;
    bool subgroup_size_control;
    bool compute_full_subgroups;
    bool has_presentation;

    // Queues (VK_NULL_HANDLE if not available). Graphics queue is always required to be available.
    VkQueue graphics_queue;
    u32 graphics_queue_family_index;
    bool graphics_queue_timestamp_queries;

    VkQueue compute_queue;
    u32 compute_queue_family_index;
    bool compute_queue_timestamp_queries;

    VkQueue transfer_queue;
    u32 transfer_queue_family_index;
    bool transfer_queue_timestamp_queries;

    // Sync command submission (maybe remove this at some point)
    VkCommandPool sync_command_pool;
    VkCommandBuffer sync_command_buffer;
    VkFence sync_fence;

    // Debug
    bool debug_utils_enabled;
};

struct DeviceDesc
{
    u32 minimum_api_version = VK_API_VERSION_1_1;

    // Check for presentation support on graphics queue and add VK_KHR_swapchain extension.
    bool require_presentation = true;

    u32 force_physical_device_index = ~0U;
    bool prefer_discrete_gpu = true;
    DeviceFeatures required_features = DeviceFeatures::NONE;
    DeviceFeatures optional_features = DeviceFeatures::NONE;
};

Result Init(bool init_glfw = true);
Result CreateInstance(Instance* instance, const InstanceDesc&& desc);
Result CreateDevice(Device* device, const Instance& instance, const DeviceDesc&& desc);
void DestroyInstance(Instance* instance);
void DestroyDevice(Device* device);

void WaitIdle(const Device& device);

//- Semaphores
#ifdef _WIN32
typedef HANDLE ExternalHandle;
#else
typedef int ExternalHandle;
#endif

VkResult GetExternalHandleForSemaphore(ExternalHandle* handle, const Device& device, VkSemaphore semaphore);
VkResult CreateGPUSemaphore(VkDevice device, VkSemaphore* semaphore, bool external = false);
VkResult CreateGPUTimelineSemaphore(VkDevice device, VkSemaphore* semaphore, uint64_t initial_value = 0, bool external = false);
void DestroyGPUSemaphore(VkDevice device, VkSemaphore* semaphore);

//- Swapchain
enum class SwapchainStatus
{
    READY,
    RESIZED,
    MINIMIZED,
    FAILED,
};

Result CreateSwapchain(Window* w, const Device& device, VkSurfaceKHR surface, VkFormat format, u32 fb_width, u32 fb_height, usize frames, VkSwapchainKHR old_swapchain);
SwapchainStatus UpdateSwapchain(Window* w, const Device& device);

//- Frame
struct Frame
{
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;

    VkCommandPool compute_command_pool;
    VkCommandBuffer compute_command_buffer;

    VkCommandPool transfer_command_pool;
    VkCommandBuffer transfer_command_buffer;

    VkSemaphore acquire_semaphore;

    VkFence fence;

    // Filled every frame after acquiring
    VkImage current_image;
    VkImageView current_image_view;
    VkSemaphore current_present_semaphore;
    u32 current_image_index;
};

Frame& WaitForFrame(Window* w, const Device& device);
Result AcquireImage(Frame* frame, Window* window, const Device& device);

//- Window
enum class MouseButton: u32 {
    None = ~0u,
    Left = GLFW_MOUSE_BUTTON_LEFT,
    Right = GLFW_MOUSE_BUTTON_RIGHT,
    Middle = GLFW_MOUSE_BUTTON_MIDDLE,

    Button1 = GLFW_MOUSE_BUTTON_1,
    Button2 = GLFW_MOUSE_BUTTON_2,
    Button3 = GLFW_MOUSE_BUTTON_3,
    Button4 = GLFW_MOUSE_BUTTON_4,
    Button5 = GLFW_MOUSE_BUTTON_5,
    Button6 = GLFW_MOUSE_BUTTON_6,
    Button7 = GLFW_MOUSE_BUTTON_7,
    Button8 = GLFW_MOUSE_BUTTON_8,
};

enum class Action: u32 {
    None = ~0u,
    Release = GLFW_RELEASE,
    Press = GLFW_PRESS,
    Repeat = GLFW_REPEAT,
};

enum class Modifiers: u32 {
    None = 0,
    Shift = GLFW_MOD_SHIFT,
    Ctrl = GLFW_MOD_CONTROL,
    Alt = GLFW_MOD_ALT,
    Super = GLFW_MOD_SUPER,
};

enum class Key: s32 {
    Space         = GLFW_KEY_SPACE,
    Apostrophe    = GLFW_KEY_APOSTROPHE,
    Comma         = GLFW_KEY_COMMA,
    Minus         = GLFW_KEY_MINUS,
    Period        = GLFW_KEY_PERIOD,
    Slash         = GLFW_KEY_SLASH,
    N0            = GLFW_KEY_0,
    N1            = GLFW_KEY_1,
    N2            = GLFW_KEY_2,
    N3            = GLFW_KEY_3,
    N4            = GLFW_KEY_4,
    N5            = GLFW_KEY_5,
    N6            = GLFW_KEY_6,
    N7            = GLFW_KEY_7,
    N8            = GLFW_KEY_8,
    N9            = GLFW_KEY_9,
    Semicolon     = GLFW_KEY_SEMICOLON,
    Equal         = GLFW_KEY_EQUAL,
    A             = GLFW_KEY_A,
    B             = GLFW_KEY_B,
    C             = GLFW_KEY_C,
    D             = GLFW_KEY_D,
    E             = GLFW_KEY_E,
    F             = GLFW_KEY_F,
    G             = GLFW_KEY_G,
    H             = GLFW_KEY_H,
    I             = GLFW_KEY_I,
    J             = GLFW_KEY_J,
    K             = GLFW_KEY_K,
    L             = GLFW_KEY_L,
    M             = GLFW_KEY_M,
    N             = GLFW_KEY_N,
    O             = GLFW_KEY_O,
    P             = GLFW_KEY_P,
    Q             = GLFW_KEY_Q,
    R             = GLFW_KEY_R,
    S             = GLFW_KEY_S,
    T             = GLFW_KEY_T,
    U             = GLFW_KEY_U,
    V             = GLFW_KEY_V,
    W             = GLFW_KEY_W,
    X             = GLFW_KEY_X,
    Y             = GLFW_KEY_Y,
    Z             = GLFW_KEY_Z,
    LeftBracket   = GLFW_KEY_LEFT_BRACKET,
    Backslash     = GLFW_KEY_BACKSLASH,
    RightBracket  = GLFW_KEY_RIGHT_BRACKET,
    GraveAccent   = GLFW_KEY_GRAVE_ACCENT,
    World1        = GLFW_KEY_WORLD_1,
    World2        = GLFW_KEY_WORLD_2,
    Escape        = GLFW_KEY_ESCAPE,
    Enter         = GLFW_KEY_ENTER,
    Tab           = GLFW_KEY_TAB,
    Backspace     = GLFW_KEY_BACKSPACE,
    Insert        = GLFW_KEY_INSERT,
    Delete        = GLFW_KEY_DELETE,
    Right         = GLFW_KEY_RIGHT,
    Left          = GLFW_KEY_LEFT,
    Down          = GLFW_KEY_DOWN,
    Up            = GLFW_KEY_UP,
    PageUp        = GLFW_KEY_PAGE_UP,
    PageDown      = GLFW_KEY_PAGE_DOWN,
    Home          = GLFW_KEY_HOME,
    End           = GLFW_KEY_END,
    CapsLock      = GLFW_KEY_CAPS_LOCK,
    ScrollLock    = GLFW_KEY_SCROLL_LOCK,
    NumLock       = GLFW_KEY_NUM_LOCK,
    PrintScreen   = GLFW_KEY_PRINT_SCREEN,
    Pause         = GLFW_KEY_PAUSE,
    F1            = GLFW_KEY_F1,
    F2            = GLFW_KEY_F2,
    F3            = GLFW_KEY_F3,
    F4            = GLFW_KEY_F4,
    F5            = GLFW_KEY_F5,
    F6            = GLFW_KEY_F6,
    F7            = GLFW_KEY_F7,
    F8            = GLFW_KEY_F8,
    F9            = GLFW_KEY_F9,
    F10           = GLFW_KEY_F10,
    F11           = GLFW_KEY_F11,
    F12           = GLFW_KEY_F12,
    F13           = GLFW_KEY_F13,
    F14           = GLFW_KEY_F14,
    F15           = GLFW_KEY_F15,
    F16           = GLFW_KEY_F16,
    F17           = GLFW_KEY_F17,
    F18           = GLFW_KEY_F18,
    F19           = GLFW_KEY_F19,
    F20           = GLFW_KEY_F20,
    F21           = GLFW_KEY_F21,
    F22           = GLFW_KEY_F22,
    F23           = GLFW_KEY_F23,
    F24           = GLFW_KEY_F24,
    F25           = GLFW_KEY_F25,
    KP0           = GLFW_KEY_KP_0,
    KP1           = GLFW_KEY_KP_1,
    KP2           = GLFW_KEY_KP_2,
    KP3           = GLFW_KEY_KP_3,
    KP4           = GLFW_KEY_KP_4,
    KP5           = GLFW_KEY_KP_5,
    KP6           = GLFW_KEY_KP_6,
    KP7           = GLFW_KEY_KP_7,
    KP8           = GLFW_KEY_KP_8,
    KP9           = GLFW_KEY_KP_9,
    KPDecimal     = GLFW_KEY_KP_DECIMAL,
    KPDivide      = GLFW_KEY_KP_DIVIDE,
    KPMultiply    = GLFW_KEY_KP_MULTIPLY,
    KPSubtract    = GLFW_KEY_KP_SUBTRACT,
    KPAdd         = GLFW_KEY_KP_ADD,
    KPEnter       = GLFW_KEY_KP_ENTER,
    KPEqual       = GLFW_KEY_KP_EQUAL,
    LeftShift     = GLFW_KEY_LEFT_SHIFT,
    LeftControl   = GLFW_KEY_LEFT_CONTROL,
    LeftAlt       = GLFW_KEY_LEFT_ALT,
    LeftSuper     = GLFW_KEY_LEFT_SUPER,
    RightShift    = GLFW_KEY_RIGHT_SHIFT,
    RightControl  = GLFW_KEY_RIGHT_CONTROL,
    RightAlt      = GLFW_KEY_RIGHT_ALT,
    RightSuper    = GLFW_KEY_RIGHT_SUPER,
    Menu          = GLFW_KEY_MENU,

    Unknown       = GLFW_KEY_UNKNOWN,
};

struct WindowCallbacks {
    Function<void(glm::ivec2)> mouse_move_event;
    Function<void(glm::ivec2, MouseButton, Action, Modifiers)> mouse_button_event;
    Function<void(glm::ivec2, glm::dvec2)> mouse_scroll_event;
    Function<void(Key, Action, Modifiers)> key_event;
    Function<void()> draw;

    // Mimic imgui chain callback mechanism, this way either we chain ImGui
    // if installed later, or ImGui chains us if installed before.
    //
    // ImGui always calls us before processing, to make the mechanism symmetric
    // regardless of installation order, we instead call imgui after processing.
    //
    // The functions commented out are callbacks that we currently do not install
    // but ImGui does. They must also be chained if we implement them.

    // Currently implemented by imgui:
    //
    // GLFWwindowfocusfun      prev_callback_window_focus;
    GLFWcursorposfun        prev_callback_cursor_pos;
    // GLFWcursorenterfun      prev_callback_cursor_enter;
    GLFWmousebuttonfun      prev_callback_mouse_button;
    GLFWscrollfun           prev_callback_scroll;
    GLFWkeyfun              prev_callback_key;
    // GLFWcharfun             prev_callback_char;
    // GLFWmonitorfun          prev_callback_monitor;

    // Not yet implemented by imgui, but by us:
    GLFWwindowrefreshfun    prev_callback_window_refresh;
};

struct StaleSwapchain
{
    // Number of frames still using this swapchain, when this number reaches 0
    // the swapchain can be safely destroyed.
    usize frames_in_flight = 0;

    // Handle to the swapchain.
    VkSwapchainKHR swapchain;

    // Image views
    Array<VkImageView> image_views;
};

struct Window
{
    GLFWwindow* window;
    VkSurfaceKHR surface;
    VkFormat swapchain_format;
    VkImageUsageFlags swapchain_usage_flags;
    VkPresentModeKHR present_mode;
    WindowCallbacks callbacks;

    // Swapchain
    u32 fb_width;
    u32 fb_height;
    VkSwapchainKHR swapchain;

    // Per frame swapchain data
    Array<VkImage> images;
    Array<VkImageView> image_views;
    Array<VkSemaphore> present_semaphores;

    // Index in swapchain frames, wraps around at the number of frames in flight.
    // This is always sequential, it should be used to index into 'frames' but not
    // into images, image_views and present semaphores.
    u32 wrapping_frame_index;

    // If set to true when calling UpdateSwapchain it will force creation of a new swapchain.
    // Should be set after a vkQueuePresentKHR returns VK_ERROR_OUT_OF_DATE_KHR or VK_SUBOPTIMAL_KHR.
    bool force_swapchain_recreate;

    // Per frame persistent data
    Array<Frame> frames;

    // List of existing swapchains, during normal operation only a single
    // swapchain exists, but when a swapchain is recreated (e.g. during
    // resizing) the previous swapchain is kept alive until all frames used
    // from it have been released.
    ObjArray<StaleSwapchain> stale_swapchains;
};

struct WindowDesc {
    const char* title;
    u32 width;
    u32 height;

    u32 x = ANY_POSITION;
    u32 y = ANY_POSITION;

    u32 preferred_frames_in_flight = 2;
    VkImageUsageFlags preferred_swapchain_usage_flags = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    bool vsync = true;
};

Result CreateWindowWithSwapchain(Window* w, const Instance& instance, const Device& device, const WindowDesc&& desc);
void DestroyWindowWithSwapchain(Window* w, const Instance& instance, const Device& device);
void CloseWindow(const Window& window);

//- Input
void Callback_WindowRefresh(GLFWwindow* window);
void Callback_MouseButton(GLFWwindow* window, int button, int action, int mods);
void Callback_Scroll(GLFWwindow* window, double x, double y);
void Callback_CursorPos(GLFWwindow* window, double x, double y);
void SetWindowCallbacks(Window* window, WindowCallbacks&& callbacks);
void ProcessEvents(bool block);
bool ShouldClose(const Window& window);
Modifiers GetModifiersState(const Window& window);

//- Queries
struct QueryPoolDesc {
    VkQueryType type;
    u32 count;
};
VkResult CreateQueryPool(VkQueryPool* pool, const Device& device, const QueryPoolDesc&& desc);
void DestroyQueryPool(VkQueryPool* pool, const Device& device);

//- Queue
struct SubmitDesc1 {
    Span<VkCommandBuffer> cmd;
    Span<VkSemaphore> wait_semaphores;
    Span<VkPipelineStageFlags> wait_stages;

    // Must contain a value for each semaphore in wait_semaphores.
    // This is only useful if one or more of these semaphores is a timeline semaphore.
    //
    // A stub value must be provided for every binary semaphore in wait_semaphores. The stub value will be ignored.
    //
    // Can be empty if no timeline semaphore is used in wait_semaphores.
    Span<u64> wait_timeline_values;

    Span<VkSemaphore> signal_semaphores;

    // Must contain a value for each semaphore in signal_semaphores.
    // This is only useful if one or more of these semaphores is a timeline semaphore.
    //
    // A stub value must be provided for every binary semaphore in signal_semaphores. The stub value will be ignored.
    //
    // Can be empty if no timeline semaphore is used in signal_semaphores.
    Span<u64> signal_timeline_values;

    VkFence fence;
};

VkResult SubmitQueue1(VkQueue queue, const SubmitDesc1&& desc);
VkResult Submit1(const Frame& frame, const Device& device, VkPipelineStageFlags wait_stage_mask);
VkResult SubmitSync1(const Device& device);

struct SemaphoreSubmitInfo {
    VkStructureType          type = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    const void*              next = nullptr;
    VkSemaphore              semaphore;
    uint64_t                 value = 0;
    VkPipelineStageFlags2    stage_mask;
    uint32_t                 device_index = 0;
};
static_assert(sizeof(SemaphoreSubmitInfo) == sizeof(VkSemaphoreSubmitInfo));

struct CommandBufferSubmitInfo {
    VkStructureType    type = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    const void*        next = nullptr;
    VkCommandBuffer    command_buffer;
    uint32_t           device_mask = 0;
};
static_assert(sizeof(CommandBufferSubmitInfo) == sizeof(CommandBufferSubmitInfo));

struct SubmitDesc {
    VkSubmitFlags flags = 0;
    Span<SemaphoreSubmitInfo> wait_semaphore_infos;
    Span<CommandBufferSubmitInfo> command_buffer_infos;
    Span<SemaphoreSubmitInfo> signal_semaphore_infos;
    VkFence fence;
};

VkResult SubmitQueue(VkQueue queue, const SubmitDesc&& desc);
VkResult Submit(const Frame& frame, const Device& device, VkPipelineStageFlags2 wait_stage_mask, VkPipelineStageFlags2 signal_stage_mask);
VkResult SubmitSync(const Device& device);

VkResult PresentFrame(Window* w, Frame* frame, const Device& device);

//- Commands
VkResult BeginCommands(VkCommandPool pool, VkCommandBuffer buffer, const Device& device);
VkResult EndCommands(VkCommandBuffer buffer);

struct MemoryBarrierDesc
{
    VkStructureType          type = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    const void*              next = nullptr;
    VkPipelineStageFlags2    src_stage;
    VkAccessFlags2           src_access;
    VkPipelineStageFlags2    dst_stage;
    VkAccessFlags2           dst_access;
};
static_assert(sizeof(MemoryBarrierDesc) == sizeof(VkMemoryBarrier2));

void CmdMemoryBarrier(VkCommandBuffer cmd, const MemoryBarrierDesc&& desc);

struct BufferBarrierDesc
{
    VkStructureType          type = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    const void*              next = nullptr;
    VkPipelineStageFlags2    src_stage;
    VkAccessFlags2           src_access;
    VkPipelineStageFlags2    dst_stage;
    VkAccessFlags2           dst_access;
    uint32_t                 src_queue = VK_QUEUE_FAMILY_IGNORED;
    uint32_t                 dst_queue = VK_QUEUE_FAMILY_IGNORED;
    VkBuffer                 buffer;
    VkDeviceSize             offset = 0;
    VkDeviceSize             size = VK_WHOLE_SIZE;
};
static_assert(sizeof(BufferBarrierDesc) == sizeof(VkBufferMemoryBarrier2));

void CmdBufferBarrier(VkCommandBuffer cmd, const BufferBarrierDesc&& desc);

struct ImageBarrierDesc
{
    VkStructureType            type = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    const void*                next = nullptr;
    VkPipelineStageFlags2      src_stage;
    VkAccessFlags2             src_access;
    VkPipelineStageFlags2      dst_stage;
    VkAccessFlags2             dst_access;
    VkImageLayout              old_layout;
    VkImageLayout              new_layout;
    uint32_t                   src_queue = VK_QUEUE_FAMILY_IGNORED;
    uint32_t                   dst_queue = VK_QUEUE_FAMILY_IGNORED;
    VkImage                    image;
    VkImageAspectFlags         aspect_mask = VK_IMAGE_ASPECT_COLOR_BIT;
    uint32_t                   base_mip_level = 0;
    uint32_t                   mip_level_count = VK_REMAINING_MIP_LEVELS;
    uint32_t                   base_array_layer = 0;
    uint32_t                   array_layer_count = VK_REMAINING_ARRAY_LAYERS;
};
static_assert(sizeof(ImageBarrierDesc) == sizeof(VkImageMemoryBarrier2));

void CmdImageBarrier(VkCommandBuffer cmd, const ImageBarrierDesc&& desc);

struct BarriersDesc {
    Span<MemoryBarrierDesc> memory;
    Span<BufferBarrierDesc> buffer;
    Span<ImageBarrierDesc> image;
};

void CmdBarriers(VkCommandBuffer cmd, const BarriersDesc&& desc);

struct RenderingAttachmentDesc {
    VkStructureType          type = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    const void*              next = nullptr;
    VkImageView              view;
    VkImageLayout            layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkResolveModeFlagBits    resolve_mode = VK_RESOLVE_MODE_NONE;
    VkImageView              resolve_image_view = VK_NULL_HANDLE;
    VkImageLayout            resolve_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkAttachmentLoadOp       load_op;
    VkAttachmentStoreOp      store_op;
    VkClearColorValue        clear;
};
static_assert(sizeof(RenderingAttachmentDesc) == sizeof(VkRenderingAttachmentInfo));

struct DepthAttachmentDesc {
    VkImageView view;
    VkAttachmentLoadOp load_op;
    VkAttachmentStoreOp store_op;
    float clear;
};

struct BeginRenderingDesc {
    Span<RenderingAttachmentDesc> color;
    DepthAttachmentDesc depth;
    // RenderingAttachmentDesc stencil;
    u32 offset_x = 0;
    u32 offset_y = 0;
    u32 width;
    u32 height;
};

void CmdBeginRendering(VkCommandBuffer cmd, const BeginRenderingDesc&& desc);
void CmdEndRendering(VkCommandBuffer cmd);


struct CopyBufferDesc {
    VkBuffer src;
    VkBuffer dst;
    VkDeviceSize src_offset = 0;
    VkDeviceSize dst_offset = 0;
    VkDeviceSize size;
};

void CmdCopyBuffer(VkCommandBuffer cmd, const CopyBufferDesc&& size);

struct CopyImageBufferDesc {
    VkImage image;
    VkImageLayout image_layout;
    u32 image_x = 0;
    u32 image_y = 0;
    u32 image_z = 0;
    u32 image_width;
    u32 image_height;
    u32 image_depth = 1;
    VkImageAspectFlags image_aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    u32 image_mip = 0;
    u32 image_base_layer = 0;  // Index of first layer in array
    u32 image_layer_count = 1; // Number of layers in array

    VkBuffer buffer;
    u64 buffer_offset_in_bytes = 0;
    u32 buffer_row_stride_in_texels = 0; // 0 means rows are tightly packed
    u32 buffer_image_height_in_texels = 0; // 0 means planes are tightly packed (for 3D images)
};
void CmdCopyImageToBuffer(VkCommandBuffer cmd, const CopyImageBufferDesc&& desc);
void CmdCopyBufferToImage(VkCommandBuffer cmd, const CopyImageBufferDesc&& desc);

struct BlitImageDesc {
    VkImage src;
    VkImageLayout src_layout;
    u32 src_x = 0;
    u32 src_y = 0;
    u32 src_z = 0;
    u32 src_width;
    u32 src_height;
    u32 src_depth = 1;
    u32 src_mip_level = 0;
    u32 src_base_array_layer = 0;  // Index of first layer in array
    u32 src_array_layer_count = 1; // Number of layers in array
    VkImageAspectFlags src_aspect = VK_IMAGE_ASPECT_COLOR_BIT;

    VkImage dst;
    VkImageLayout dst_layout;
    u32 dst_x = 0;
    u32 dst_y = 0;
    u32 dst_z = 0;
    u32 dst_width;
    u32 dst_height;
    u32 dst_depth = 1;
    u32 dst_mip_level = 0;
    u32 dst_base_array_layer = 0;  // Index of first layer in array
    u32 dst_array_layer_count = 1; // Number of layers in array
    VkImageAspectFlags dst_aspect = VK_IMAGE_ASPECT_COLOR_BIT;

    VkFilter filter;
};
void CmdBlitImage(VkCommandBuffer cmd, const BlitImageDesc&& desc);

struct ResolveImageDesc {
    VkImage src;
    VkImageLayout src_layout;
    u32 src_x = 0;
    u32 src_y = 0;
    u32 src_z = 0;
    u32 src_mip = 0;
    u32 src_base_layer = 0;  // Index of first layer in array
    u32 src_layer_count = 1; // Number of layers in array
    VkImageAspectFlags src_aspect = VK_IMAGE_ASPECT_COLOR_BIT;

    VkImage dst;
    VkImageLayout dst_layout;
    u32 dst_x = 0;
    u32 dst_y = 0;
    u32 dst_z = 0;
    u32 dst_mip = 0;
    u32 dst_base_layer = 0;  // Index of first layer in array
    u32 dst_layer_count = 1; // Number of layers in array
    VkImageAspectFlags dst_aspect = VK_IMAGE_ASPECT_COLOR_BIT;

    u32 width;
    u32 height;
    u32 depth = 1;
};
void CmdResolveImage(VkCommandBuffer cmd, const ResolveImageDesc&& desc);

void CmdDrawMeshTasks(VkCommandBuffer cmd, u32 groups_x, u32 groups_y, u32 groups_z);
void CmdDrawMeshTasksIndirect(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset, u32 draw_count, u32 stride);
void CmdDrawMeshTasksIndirectCount(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset, VkBuffer count_buffer, VkDeviceSize count_offset, u32 max_draw_count, u32 stride);


// - Shaders
struct Shader {
    VkShaderModule shader;
};

VkResult CreateShader(Shader* shader, const Device& device, ArrayView<u8> code);
void DestroyShader(Shader* shader, const Device& device);


//- Pipelines
struct GraphicsPipeline {
    VkPipeline pipeline;
    VkPipelineLayout layout;
};

struct PipelineStageDesc {
    Shader shader;
    VkShaderStageFlagBits stage;
    const char* entry = "main";
    u32 required_subgroup_size = 0;
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
    VkPrimitiveTopology primitive_topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    bool primitive_restart_enable = false;
};

struct RasterizationDesc {
    VkPolygonMode polygon_mode = VK_POLYGON_MODE_FILL;
    VkCullModeFlags cull_mode = VK_CULL_MODE_NONE;
    VkFrontFace front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    bool depth_bias_enable = false;
    bool depth_clamp_enable = false;
    bool dynamic_line_width = false;
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
struct PushConstantsRangeDesc {
    VkShaderStageFlags flags = VK_SHADER_STAGE_ALL;
    u32 offset;
    u32 size;
};

struct GraphicsPipelineDesc {
    Span<PipelineStageDesc> stages;
    Span<VertexBindingDesc> vertex_bindings;
    Span<VertexAttributeDesc> vertex_attributes;
    InputAssemblyDesc input_assembly;
    RasterizationDesc rasterization;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    DepthDesc depth;
    StencilDesc stencil;
    Span<PushConstantsRangeDesc> push_constants;
    Span<VkDescriptorSetLayout> descriptor_sets;
    Span<AttachmentDesc> attachments;
};

VkResult CreateGraphicsPipeline(GraphicsPipeline* graphics_pipeline, const Device& device, const GraphicsPipelineDesc&& desc);
void DestroyGraphicsPipeline(GraphicsPipeline* pipeline, const Device& device);

struct ComputePipeline {
    VkPipeline pipeline;
    VkPipelineLayout layout;
};

struct ComputePipelineDesc {
    Shader shader;
    const char* entry = "main";
    Span<PushConstantsRangeDesc> push_constants;
    Span<VkDescriptorSetLayout> descriptor_sets;
    u32 required_subgroup_size = 0;
};

VkResult CreateComputePipeline(ComputePipeline* compute_pipeline, const Device& device, const ComputePipelineDesc&& desc);
void DestroyComputePipeline(ComputePipeline* pipeline, const Device& device);

//- Allocation info
struct AllocDesc {
    VmaMemoryUsage vma_usage = VMA_MEMORY_USAGE_AUTO;
    VmaAllocationCreateFlags vma_flags = 0;

    // Necessary when using VMA_MEMORY_USAGE_UNKNOWN, otherwise optional
    VkMemoryPropertyFlags memory_properties_required = 0;
    VkMemoryPropertyFlags memory_properties_preferred = 0;
};

// TODO: should technically remember to invalidate/flush these if the allocations are mapped but not coherent,
// but no GPU that we know of has non-coherent memory on Desktop so for now we ignore this.
namespace AllocPresets {
    constexpr AllocDesc Host = {
        .vma_usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .vma_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
    };

    constexpr AllocDesc HostWriteCombining = {
        .vma_usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .vma_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
    };

    // Must manually check if the allocation is visible and potentially allocate a staging buffer if it did not
    // go to BAR visible memory.
    constexpr AllocDesc DeviceMappedWithFallback = {
        .vma_usage = VMA_MEMORY_USAGE_AUTO,
        .vma_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT |  VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT,
    };

    // Falls back to host if device bar not available
    constexpr AllocDesc DeviceMapped = {
        .vma_usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .vma_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
    };

    constexpr AllocDesc Device = {
        .vma_usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    };

    constexpr AllocDesc DeviceDedicated = {
        .vma_usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .vma_flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
    };

    enum class Type {
        Host,
        HostWriteCombining,
        DeviceMappedWithFallback,
        DeviceMapped,
        Device,
        DeviceDedicated,
    };

    constexpr AllocDesc Types[] = {
        Host,
        HostWriteCombining,
        DeviceMappedWithFallback,
        DeviceMapped,
        Device,
        DeviceDedicated,
    };
}

//- Buffers
struct Buffer
{
    VkBuffer buffer;
    VmaAllocation allocation;
    ArrayView<u8> map;
};

struct BufferDesc {
    // Vulkan flags
    VkBufferUsageFlags usage;

    // Allocation info
    AllocDesc alloc;

    // Pool
    VmaPool pool = VK_NULL_HANDLE;
    bool external = false;
};

VkResult CreateBuffer(Buffer* buffer, const Device& device, size_t size, const BufferDesc&& desc);
VkResult CreateBufferFromData(Buffer* buffer, const Device& device, ArrayView<u8> data, const BufferDesc&& desc);
void DestroyBuffer(Buffer* buffer, const Device& device);

struct Pool {
    VmaPool pool;
    VkExportMemoryAllocateInfo* export_mem_alloc_info = nullptr;
};

struct PoolBufferDesc {
    // Vulkan flags
    VkBufferUsageFlags usage;

    // Allocation info
    AllocDesc alloc;

    // Allow external usage
    bool external = false;
};

VkResult CreatePoolForBuffer(Pool* pool, const Device& device, const PoolBufferDesc&& desc);
void DestroyPool(Pool* pool, const Device& device);

VkResult GetExternalHandleForBuffer(ExternalHandle* handle, const Device& device, const Buffer& buffer);
void CloseExternalHandle(ExternalHandle* handle);

// - Images
struct Image
{
    VkImage image;
    VkImageView view;
    VmaAllocation allocation;
};

struct ImageDesc {
    u32 width;
    u32 height;
    u32 depth = 1;
    VkFormat format;
    u32 mip_levels = 1;
    u32 array_layers = 1;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    VkImageCreateFlags flags = 0;

    // Image view
    VkImageViewType image_view_type = VK_IMAGE_VIEW_TYPE_2D;

    // Vulkan flags
    VkImageUsageFlags usage;

    // VMA flags
    AllocDesc alloc;
};

VkResult CreateImage(Image* image, const Device& device, const ImageDesc&& desc);

struct ImageUploadDesc {
    u32 width;
    u32 height;
    u32 depth = 1;
    VkFormat format;
    VkImageLayout current_image_layout;
    VkImageLayout final_image_layout;
};

VkResult UploadImage(const Image& image, const Device& device, ArrayView<u8> data, const ImageUploadDesc&& desc);

VkResult CreateAndUploadImage(Image* image, const Device& device, ArrayView<u8> data, VkImageLayout layout, const ImageDesc&& desc);
void DestroyImage(Image* image, const Device& device);

//- Sampler
struct Sampler {
    VkSampler sampler;
};

struct SamplerDesc {
    VkFilter min_filter = VK_FILTER_NEAREST;
    VkFilter mag_filter = VK_FILTER_NEAREST;
    VkSamplerMipmapMode mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    float mip_lod_bias = 0.0f;
    float min_lod = 0.0f;
    float max_lod = VK_LOD_CLAMP_NONE;

    VkSamplerAddressMode u = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    VkSamplerAddressMode v = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    VkSamplerAddressMode w = VK_SAMPLER_ADDRESS_MODE_REPEAT;

    bool anisotroy_enabled = false;
    float max_anisotropy = 0.0f;

    bool compare_enable = false;
    VkCompareOp compare_op = VK_COMPARE_OP_ALWAYS;

    VkBorderColor border_color = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
};

VkResult CreateSampler(Sampler* sampler, const Device& device, const SamplerDesc&& desc);
void DestroySampler(Sampler* sampler, const Device& device);

//- Descriptors
struct DescriptorSetLayout
{
    VkDescriptorSetLayout layout;
};

struct DescriptorSetBindingDesc
{
    u32                      count;
    VkDescriptorType         type;
    VkShaderStageFlags       stage_flags = VK_SHADER_STAGE_ALL;
    ArrayView<VkSampler>     immutable_samplers;
    VkDescriptorBindingFlags flags = 0;
};

struct DescriptorSetLayoutDesc
{
    Span<DescriptorSetBindingDesc> bindings;
    VkDescriptorSetLayoutCreateFlags flags = 0;
};

VkResult CreateDescriptorSetLayout(DescriptorSetLayout* layout, const Device& device, const DescriptorSetLayoutDesc&& desc);
void DestroyDescriptorSetLayout(DescriptorSetLayout* layout, const Device& device);

struct DescriptorPool
{
    VkDescriptorPool pool;
};

struct DescriptorPoolDesc
{
    Span<VkDescriptorPoolSize> sizes;
    u32 max_sets;
    VkDescriptorPoolCreateFlags flags = 0;
};

VkResult CreateDescriptorPool(DescriptorPool* pool, const Device& device, const DescriptorPoolDesc&& desc);
VkResult ResetDescriptorPool(DescriptorPool pool, const Device& device);
void DestroyDescriptorPool(DescriptorPool* pool, const Device& device);

struct DescriptorSet
{
    VkDescriptorSet set;
};

struct DescriptorSetAllocDesc
{
    DescriptorPool pool;
    DescriptorSetLayout layout;
    u32 variable_size_count = 0;
};

VkResult AllocateDescriptorSet(DescriptorSet* set, const Device& device, const DescriptorSetAllocDesc&& desc);
VkResult FreeDescriptorSet(DescriptorPool pool, DescriptorSet set, const Device& device);

struct DescriptorPoolLayoutAndSet
{
    DescriptorPool pool;
    DescriptorSetLayout layout;
    DescriptorSet set;
};

struct DescriptorPoolLayoutAndSetDesc
{
    // Layout
    Span<DescriptorSetBindingDesc> bindings;
    VkDescriptorSetLayoutCreateFlags layout_flags = 0;

    // Pool
    VkDescriptorPoolCreateFlags pool_flags = 0;
};

VkResult CreateDescriptorPoolLayoutAndSet(DescriptorPoolLayoutAndSet* pool_layout_set, const Device& device, const DescriptorPoolLayoutAndSetDesc&& desc);
void DestroyDescriptorPoolLayoutAndSet(DescriptorPoolLayoutAndSet* pool_layout_set, const Device& device);


//- Descriptor writes
struct BufferDescriptorWriteDesc {
    VkBuffer buffer;
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    u32 binding;
    u32 element = 0;
    VkDeviceSize offset = 0;
    VkDeviceSize size = VK_WHOLE_SIZE;
};

struct ImageDescriptorWriteDesc {
    VkImageView view;
    VkImageLayout layout;

    VkDescriptorType type;
    u32 binding;
    u32 element = 0;
};

struct CombinedImageSamplerDescriptorWriteDesc {
    VkImageView view;
    VkImageLayout layout;
    VkSampler sampler;

    u32 binding;
    u32 element = 0;
};

struct SamplerDescriptorWriteDesc {
    VkSampler sampler;
    u32 binding;
    u32 element = 0;
};

struct AccelerationStructureDescriptorWriteDesc {
    VkAccelerationStructureKHR acceleration_structure;
    u32 binding;
    u32 element = 0;
};

void WriteBufferDescriptor(DescriptorSet set, const Device& device, const BufferDescriptorWriteDesc&& write);
void WriteImageDescriptor(DescriptorSet set, const Device& device, const ImageDescriptorWriteDesc&& write);
void WriteSamplerDescriptor(DescriptorSet set, const Device& device, const SamplerDescriptorWriteDesc&& write);
void WriteCombinedImageSamplerDescriptor(DescriptorSet set, const Device& device, const CombinedImageSamplerDescriptorWriteDesc&& write);
void WriteAccelerationStructureDescriptor(DescriptorSet set, const Device& device, const AccelerationStructureDescriptorWriteDesc&& write);

// Acceleration structures


struct AccelerationStructure {
    Array<VkAccelerationStructureKHR> blas;
    VkAccelerationStructureKHR tlas;
    Buffer blas_buffer;
    Buffer tlas_buffer;
    Buffer instances_buffer;
};

struct AccelerationStructureMeshDesc {
    VkDeviceAddress vertices_address;
    u64 vertices_stride;
    u32 vertices_count;
    VkFormat vertices_format;

    VkDeviceAddress indices_address;
    VkIndexType indices_type;
    u32 primitive_count;

    glm::mat4x3 transform;

    u8 instance_mask = 0xFF;
};

struct AccelerationStructureDesc {
    Span<AccelerationStructureMeshDesc> meshes;
    bool prefer_fast_build = false; // If false, prefers fast trace
};

VkDeviceAddress GetBufferAddress(VkBuffer buffer, VkDevice device);
VkResult CreateAccelerationStructure(AccelerationStructure* as, const Device& device, const AccelerationStructureDesc&& desc);
void DestroyAccelerationStructure(AccelerationStructure* as, const Device& device);

//- Formats

// IMPORTANT: do not reorder those fields without updating GetFormatInfo()
struct FormatInfo {
    u32 size;
    u32 channels;
    u32 size_of_block_in_bytes;
    u32 block_side_in_pixels;
};
FormatInfo GetFormatInfo(VkFormat format);

} // namespace gfx
} // namespace xpg
