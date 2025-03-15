#pragma once

// External
#include <volk.h>                           // Vulkan
#include <vulkan/vk_enum_string_helper.h>   // Vulkan helper strings for printing

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
#include <xpg/function.h>
#include <xpg/log.h>
#include <xpg/platform.h>
#include <xpg/array.h>

#define XPG_VERSION 0

namespace gfx {

// Missing helpers:
// [ ] commands
//      [x] Dynamic rendering
//      [ ] Dispatches
//      [ ] Copies
// [ ] command buffer / pools creation (duplicate for sync and per frame)
// [ ] descriptors:
//     [ ] normal set creation
//     [ ] writes
//     [ ] bindless descriptors management helpers

struct Window;

enum class Result
{
    SUCCESS,

    VULKAN_NOT_SUPPORTED,
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

// TODO: substitute this with more extendable logic once
// we rework device picking logic.
struct DeviceFeatures {
    enum DeviceFeaturesFlags {
        NONE                  = 0,
        DESCRIPTOR_INDEXING   = 1 << 0,
        DYNAMIC_RENDERING     = 1 << 1,
        SYNCHRONIZATION_2     = 1 << 2,
        SCALAR_BLOCK_LAYOUT   = 1 << 3,
        RAY_QUERY             = 1 << 4,
        RAY_PIPELINE          = 1 << 5,
    };

    DeviceFeatures(DeviceFeaturesFlags flags): flags(flags) {}
    DeviceFeatures(uint64_t flags): flags((DeviceFeaturesFlags)flags) {}

    operator bool() {
        return (uint64_t)flags != 0;
    }

    DeviceFeatures operator|(const DeviceFeaturesFlags& b) const {
        return (DeviceFeaturesFlags)((uint64_t)flags | (uint64_t)b);
    }

    DeviceFeatures operator|(const DeviceFeatures& b) const {
        return (DeviceFeaturesFlags)((uint64_t)flags | (uint64_t)b.flags);
    }

    DeviceFeatures operator&(const DeviceFeaturesFlags& b) const {
        return (DeviceFeaturesFlags)((uint64_t)flags & (uint64_t)b);
    }

    DeviceFeatures operator&(const DeviceFeatures& b) const {
        return (DeviceFeaturesFlags)((uint64_t)flags & (uint64_t)b.flags);
    }

    DeviceFeaturesFlags flags;
};

//- Context
struct Context
{
    u32 version;
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    u32 queue_family_index;
    VkQueue copy_queue;
    u32 copy_queue_family_index;
    VmaAllocator vma;
    uint32_t preferred_frames_in_flight = 2;

    // Sync command submission
    VkCommandPool sync_command_pool;
    VkCommandBuffer sync_command_buffer;
    VkFence sync_fence;

    // Debug
    VkDebugReportCallbackEXT debug_callback;
};

struct ContextDesc {
    u32 minimum_api_version;
    ArrayView<const char *> instance_extensions;
    ArrayView<const char *> device_extensions;
    DeviceFeatures device_features;
    bool enable_validation_layer = false;
    bool enable_gpu_based_validation = false;
    uint32_t preferred_frames_in_flight = 2;
};

Result Init();
Array<const char*> GetPresentationInstanceExtensions();
Result CreateContext(Context* vk, const ContextDesc&& desc);
void DestroyContext(Context* vk);
void WaitIdle(Context& vk);
VkResult CreateGPUSemaphore(VkDevice device, VkSemaphore* semaphore);
void DestroyGPUSemaphore(VkDevice device, VkSemaphore semaphore);

//- Swapchain
enum class SwapchainStatus
{
    READY,
    RESIZED,
    MINIMIZED,
    FAILED,
};

Result CreateSwapchain(Window* w, const Context& vk, VkSurfaceKHR surface, VkFormat format, u32 fb_width, u32 fb_height, usize frames, VkSwapchainKHR old_swapchain);
SwapchainStatus UpdateSwapchain(Window* w, const Context& vk);

//- Frame
struct Frame
{
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;

    VkCommandPool copy_command_pool;
    VkCommandBuffer copy_command_buffer;

    VkSemaphore acquire_semaphore;
    VkSemaphore release_semaphore;

    VkFence fence;

    // Filled every frame after acquiring
    VkImage current_image;
    VkImageView current_image_view;
    u32 current_image_index;
};

Frame& WaitForFrame(Window* w, const Context& vk);
Result AcquireImage(Frame* frame, Window* window, const Context& vk);
Frame* AcquireNextFrame(Window* w, const Context& vk);

//- Window
enum class MouseButton: u32 {
    None = ~0u,
    Left = GLFW_MOUSE_BUTTON_LEFT,
    Right = GLFW_MOUSE_BUTTON_RIGHT,
    Middle = GLFW_MOUSE_BUTTON_MIDDLE,
};

enum class Action: u32 {
    None = ~0u,
    Release = GLFW_RELEASE,
    Press = GLFW_PRESS,
    Repeat = GLFW_REPEAT,
};

enum class Modifiers: u32 {
    Shift = GLFW_MOD_SHIFT,
    Ctrl = GLFW_MOD_CONTROL,
    Alt = GLFW_MOD_ALT,
    Super = GLFW_MOD_SUPER,
};

enum class Key: u32 {
    Escape = GLFW_KEY_ESCAPE,
    Space = GLFW_KEY_SPACE,
    Period = GLFW_KEY_PERIOD,
    Comma = GLFW_KEY_COMMA,
};

struct WindowCallbacks {
    Function<void(glm::ivec2)> mouse_move_event;
    Function<void(glm::ivec2, MouseButton, Action, Modifiers)> mouse_button_event;
    Function<void(glm::ivec2, glm::ivec2)> mouse_scroll_event;
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
    WindowCallbacks callbacks;

    // Swapchain
    u32 fb_width;
    u32 fb_height;
    VkSwapchainKHR swapchain;

    // Per frame swapchain data
    Array<VkImage> images;
    Array<VkImageView> image_views;

    // Index in swapchain frames, wraps around at the number of frames in flight.
    // This is always sequential, it should be used to index into 'frames' but not
    // into images and image_views. For that one should use the index returned
    // by vkAcquireNextImageKHR.
    u32 swapchain_frame_index;

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

Result CreateWindowWithSwapchain(Window* w, const Context& vk, const char* name, u32 width, u32 height);
void DestroyWindowWithSwapchain(Window* w, const Context& vk);
void CloseWindow(const Window& window);

//- Input
void Callback_WindowRefresh(GLFWwindow* window);
void Callback_MouseButton(GLFWwindow* window, int button, int action, int mods);
void Callback_Scroll(GLFWwindow* window, double x, double y);
void Callback_CursorPos(GLFWwindow* window, double x, double y);
void SetWindowCallbacks(Window* window, WindowCallbacks&& callbacks);
void ProcessEvents(bool block);
bool ShouldClose(const Window& window);

//- Queue
VkResult Submit(const Frame& frame, const Context& vk, VkSubmitFlags submit_stage_mask);
VkResult SubmitSync(const Context& vk);
VkResult PresentFrame(Window* w, Frame* frame, const Context& vk);

//- Commands
VkResult BeginCommands(VkCommandPool pool, VkCommandBuffer buffer, const Context& vk);
VkResult EndCommands(VkCommandBuffer buffer);

struct MemoryBarrierDesc
{
    VkPipelineStageFlags2 src_stage;
    VkPipelineStageFlags2 dst_stage;
    VkAccessFlags2 src_access;
    VkAccessFlags2 dst_access;
};

void CmdMemoryBarrier(VkCommandBuffer cmd, const MemoryBarrierDesc&& desc);

struct BufferBarrierDesc
{
    VkBuffer buffer;
    VkPipelineStageFlags2 src_stage;
    VkPipelineStageFlags2 dst_stage;
    VkAccessFlags2 src_access;
    VkAccessFlags2 dst_access;
    u32 src_queue = VK_QUEUE_FAMILY_IGNORED;
    u32 dst_queue = VK_QUEUE_FAMILY_IGNORED;
    u64 offset = 0;
    u64 size = VK_WHOLE_SIZE;
};

void CmdBufferBarrier(VkCommandBuffer cmd, const BufferBarrierDesc&& desc);

struct ImageBarrierDesc
{
    VkImage image;
    VkPipelineStageFlags2 src_stage;
    VkPipelineStageFlags2 dst_stage;
    VkAccessFlags2 src_access;
    VkAccessFlags2 dst_access;
    VkImageLayout old_layout;
    VkImageLayout new_layout;
    VkImageAspectFlags aspect_mask = VK_IMAGE_ASPECT_COLOR_BIT;
};

void CmdImageBarrier(VkCommandBuffer cmd, const ImageBarrierDesc&& desc);

struct BarriersDesc {
    Span<MemoryBarrierDesc> memory;
    Span<ImageBarrierDesc> image;
};

void CmdBarriers(VkCommandBuffer cmd, const BarriersDesc&& desc);

struct RenderingAttachmentDesc {
    VkImageView view;
    VkAttachmentLoadOp load_op;
    VkAttachmentStoreOp store_op;
    VkClearColorValue clear;
    VkResolveModeFlagBits resolve_mode = VK_RESOLVE_MODE_NONE;
    VkImageView resolve_image_view = VK_NULL_HANDLE;
    VkImageLayout resolve_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
};

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

// - Shaders
struct Shader {
    VkShaderModule shader;
};

VkResult CreateShader(Shader* shader, const Context& vk, ArrayView<u8> code);
void DestroyShader(Shader* shader, const Context& vk);


//- Pipelines
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
    VkPrimitiveTopology primitive_topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    bool primitive_restart_enable = false;
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

VkResult CreateGraphicsPipeline(GraphicsPipeline* graphics_pipeline, const Context& vk, const GraphicsPipelineDesc&& desc);
void DestroyGraphicsPipeline(GraphicsPipeline* pipeline, const Context& vk);

struct ComputePipeline {
    VkPipeline pipeline;
    VkPipelineLayout layout;
};

struct ComputePipelineDesc {
    Shader shader;
    const char* entry = "main";
    Span<PushConstantsRangeDesc> push_constants;
    Span<VkDescriptorSetLayout> descriptor_sets;
};

VkResult CreateComputePipeline(ComputePipeline* compute_pipeline, const Context& vk, const ComputePipelineDesc&& desc);
void DestroyComputePipeline(ComputePipeline* pipeline, const Context& vk);

//- Allocation info
struct AllocDesc {
    VmaMemoryUsage vma_usage = VMA_MEMORY_USAGE_AUTO;
    VmaAllocationCreateFlags vma_flags = 0;

    // Necessary when using VMA_MEMORY_USAGE_UNKNOWN, otherwise optional
    VkMemoryPropertyFlags memory_properties_required = 0;
    VkMemoryPropertyFlags memory_properties_preferred = 0;
};

// TODO: need to add host coherent required flag to most of these, otherwise need to optionally
// extra flushes that we are generally skipping.
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
};

VkResult CreateBuffer(Buffer* buffer, const Context& vk, size_t size, const BufferDesc&& desc);
VkResult CreateBufferFromData(Buffer* buffer, const Context& vk, ArrayView<u8> data, const BufferDesc&& desc);
void DestroyBuffer(Buffer* buffer, const Context& vk);

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
    VkFormat format;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;

    // Vulkan flags
    VkImageUsageFlags usage;

    // VMA flags
    AllocDesc alloc;
};

VkResult CreateImage(Image* image, const Context& vk, const ImageDesc&& desc);

struct ImageUploadDesc {
    u32 width;
    u32 height;
    VkFormat format;
    VkImageLayout current_image_layout;
    VkImageLayout final_image_layout;
};

VkResult UploadImage(const Image& image, const Context& vk, ArrayView<u8> data, const ImageUploadDesc&& desc);

VkResult CreateAndUploadImage(Image* image, const Context& vk, ArrayView<u8> data, VkImageLayout layout, const ImageDesc&& desc);
void DestroyImage(Image* image, const Context& vk);

//- Depth (todo: add desc)
struct DepthBuffer
{
    VkImage image;
    VkImageView view;
    VmaAllocation allocation;
};

VkResult CreateDepthBuffer(DepthBuffer* depth_buffer, const Context& vk, u32 width, u32 height);
void DestroyDepthBuffer(DepthBuffer* depth_buffer, const Context& vk);

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
};

VkResult CreateSampler(Sampler* sampler, const Context& vk, const SamplerDesc&& desc);
void DestroySampler(Sampler* sampler, const Context& vk);

//- Descriptors
struct DescriptorSet
{
    VkDescriptorSet set;
    VkDescriptorSetLayout layout;
    VkDescriptorPool pool;
};

struct DescriptorSetEntryDesc
{
    u32 count;
    VkDescriptorType type;
};

struct DescriptorSetDesc
{
    Span<DescriptorSetEntryDesc> entries;
    VkDescriptorBindingFlags flags;
};

VkResult CreateDescriptorSet(DescriptorSet* set, const Context& vk, const DescriptorSetDesc&& desc);
void DestroyDescriptorSet(DescriptorSet* bindless, const Context& vk);


//- Descriptor writes
struct BufferDescriptorWriteDesc {
    VkBuffer buffer;
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    u32 binding;
    u32 element = 0;
};

struct ImageDescriptorWriteDesc {
    VkImageView view;
    VkImageLayout layout;

    VkDescriptorType type;
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

void WriteBufferDescriptor(VkDescriptorSet set, const Context& vk, const BufferDescriptorWriteDesc&& write);
void WriteImageDescriptor(VkDescriptorSet set, const Context& vk, const ImageDescriptorWriteDesc&& write);
void WriteSamplerDescriptor(VkDescriptorSet set, const Context& vk, const SamplerDescriptorWriteDesc&& write);
void WriteAccelerationStructureDescriptor(VkDescriptorSet set, const Context& vk, const AccelerationStructureDescriptorWriteDesc&& write);

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
};

struct AccelerationStructureDesc {
    Span<AccelerationStructureMeshDesc> meshes;
    // TODO: fast build / compact flags
};
 
VkDeviceAddress GetBufferAddress(VkBuffer buffer, VkDevice device);
VkResult CreateAccelerationStructure(AccelerationStructure* as, const Context& vk, const AccelerationStructureDesc&& desc);
void DestroyAccelerationStructure(AccelerationStructure* as, const Context& vk);

//- Formats

// IMPORTANT: do not reorder those fields without updating GetFormatInfo()
struct FormatInfo {
    u32 size;
    u32 channels;
};
FormatInfo GetFormatInfo(VkFormat format);

}
