#include <array>
#include <vector>
#include <optional>
#include <memory>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/ndarray.h>

#include <nanobind/intrusive/counter.h>
#include <nanobind/intrusive/ref.h>

#include <xpg/gui.h>
#include <xpg/log.h>

#include "py_function.h"

namespace nb = nanobind;
using namespace xpg;

#define DEBUG_UTILS_OBJECT_NAME_WITH_NAME(type, obj, name) \
    if(obj && ctx->vk.debug_utils_enabled && (name).has_value()) { \
        VkDebugUtilsObjectNameInfoEXT name_info = { VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT }; \
        name_info.objectType = type; \
        name_info.objectHandle = (u64)obj; \
        name_info.pObjectName = (name).value().c_str(); \
        vkSetDebugUtilsObjectNameEXT(ctx->vk.device, &name_info); \
    }

#define DEBUG_UTILS_OBJECT_NAME(type, obj) DEBUG_UTILS_OBJECT_NAME_WITH_NAME(type, obj, this->name)

struct MemoryHeap: nb::intrusive_base {
    MemoryHeap(const VkMemoryHeap& heap)
        : size(heap.size)
        , flags((VkMemoryHeapFlagBits)heap.flags)
    { }
    VkDeviceSize         size;
    VkMemoryHeapFlagBits    flags;
};

struct MemoryType: nb::intrusive_base {
    MemoryType(const VkMemoryType& type)
        : property_flags((VkMemoryPropertyFlagBits)type.propertyFlags)
        , heap_index(type.heapIndex)
    { }
    VkMemoryPropertyFlagBits    property_flags;
    uint32_t                 heap_index;
};

struct MemoryProperties: nb::intrusive_base {
    MemoryProperties(const VkPhysicalDeviceMemoryProperties& memory_properties) {
        for(usize i = 0; i < memory_properties.memoryHeapCount; i++) {
            memory_heaps.push_back(new MemoryHeap(memory_properties.memoryHeaps[i]));
        }
        for(usize i = 0; i < memory_properties.memoryTypeCount; i++) {
            memory_types.push_back(new MemoryType(memory_properties.memoryTypes[i]));
        }
    };
    std::vector<nb::ref<MemoryType>> memory_types;
    std::vector<nb::ref<MemoryHeap>> memory_heaps;
};

// Wrapper around VkPhysicalDeviceLimits
struct DeviceSparseProperties: nb::intrusive_base {
    DeviceSparseProperties(const VkPhysicalDeviceSparseProperties& sparse_properties): sparse_properties(sparse_properties) {}
    VkPhysicalDeviceSparseProperties sparse_properties;
};

struct DeviceLimits: nb::intrusive_base {
    DeviceLimits(const VkPhysicalDeviceLimits& limits): limits(limits) {}
    VkPhysicalDeviceLimits limits;
};

struct HeapStatistics: nb::intrusive_base {
    HeapStatistics(const VmaBudget& budget)
        : block_count(budget.statistics.blockCount)
        , allocation_count(budget.statistics.allocationCount)
        , block_bytes(budget.statistics.blockBytes)
        , allocation_bytes(budget.statistics.allocationBytes)
        , usage(budget.usage)
        , budget(budget.budget)
    {}

    // See VmaStatistics
    u32 block_count;
    u32 allocation_count;
    u32 block_bytes;
    u32 allocation_bytes;

    // See VmaBudget
    VkDeviceSize usage;
    VkDeviceSize budget;
};

// Wrapper around VkPhysicalDeviceProperties
struct DeviceProperties: nb::intrusive_base {
    DeviceProperties(const VkPhysicalDeviceProperties& properties)
        : api_version(properties.apiVersion)
        , driver_version(properties.driverVersion)
        , vendor_id(properties.vendorID)
        , device_id(properties.deviceID)
        , device_type(properties.deviceType)
    {
        memcpy(device_name, properties.deviceName, sizeof(device_name));
        memcpy(pipeline_cache_uuid, properties.pipelineCacheUUID, sizeof(pipeline_cache_uuid));
        limits = new DeviceLimits(properties.limits);
        sparse_properties = new DeviceSparseProperties(properties.sparseProperties);
    }

    uint32_t api_version;
    uint32_t driver_version;
    uint32_t vendor_id;
    uint32_t device_id;
    VkPhysicalDeviceType device_type;
    char device_name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
    uint8_t pipeline_cache_uuid[VK_UUID_SIZE];
    nb::ref<DeviceLimits> limits;
    nb::ref<DeviceSparseProperties> sparse_properties;
};

struct Context: nb::intrusive_base {
    Context(
        std::tuple<u32, u32> version,
        gfx::DeviceFeatures::Flags required_features,
        gfx::DeviceFeatures::Flags optional_features,
        bool presentation,
        u32 preferred_frames_in_flight,
        bool vsync,
        u32 force_physical_device_index,
        bool prefer_discrete_gpu,
        bool enable_debug_utils,
        bool enable_validation_layer,
        bool enable_gpu_based_validation,
        bool enable_synchronization_validation
    )
    {
        gfx::Result result;
        result = gfx::Init();
        if (result != gfx::Result::SUCCESS) {
            throw std::runtime_error("Failed to initialize platform");
        }

        result = gfx::CreateContext(&vk, {
            .minimum_api_version = VK_MAKE_API_VERSION(0, std::get<0>(version), std::get<1>(version), 0),
            .force_physical_device_index = force_physical_device_index,
            .prefer_discrete_gpu = prefer_discrete_gpu,
            .required_features = required_features,
            .optional_features = optional_features,
            .require_presentation = presentation,
            .preferred_frames_in_flight = preferred_frames_in_flight,
            .vsync = vsync,
            .enable_debug_utils = enable_debug_utils,
            .enable_validation_layer = enable_validation_layer,
            .enable_gpu_based_validation = enable_gpu_based_validation,
            .enable_synchronization_validation = enable_synchronization_validation,
        });

        if (result != gfx::Result::SUCCESS) {
            throw std::runtime_error("Failed to initialize vulkan");
        }

        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(vk.physical_device, &properties);
        device_properties = new DeviceProperties(properties);

        VkPhysicalDeviceMemoryProperties vk_memory_properties;
        vkGetPhysicalDeviceMemoryProperties(vk.physical_device, &vk_memory_properties);
        memory_properties = new MemoryProperties(vk_memory_properties);
    }

    ~Context() {
        gfx::WaitIdle(vk);
        gfx::DestroyContext(&vk);
        logging::info("gfx", "done");
    }

    gfx::Context vk;
    nb::ref<DeviceProperties> device_properties;
    nb::ref<MemoryProperties> memory_properties;
};

struct GfxObject: nb::intrusive_base {
    GfxObject() {}
    GfxObject(nb::ref<Context> ctx, bool owned, std::optional<nb::str> name = std::nullopt)
        : ctx(std::move(ctx))
        , owned(owned)
        , name(std::move(name))
    {}

    // Reference to main context
    nb::ref<Context> ctx;

    // If set the underlying object should be freed on destruction.
    // User created objects normally have this set to true,
    // context/swapchain owned objects have this set to false.
    bool owned = true;

    // Debug name, used in __repr__ and set for vkSetDebugUtilsObjectNameEXT
    std::optional<nb::str> name;
};

struct AllocInfo: nb::intrusive_base {
    AllocInfo(const VmaAllocationInfo2& info)
        : memory_type(info.allocationInfo.memoryType)
        , offset(info.allocationInfo.offset)
        , size(info.allocationInfo.size)
        , is_dedicated(info.dedicatedMemory)
    { }

    u32 memory_type;
    VkDeviceSize offset;
    VkDeviceSize size;
    bool is_dedicated;
};

struct Buffer: GfxObject {
    Buffer(nb::ref<Context> ctx, usize size, VkBufferUsageFlagBits usage_flags, gfx::AllocPresets::Type alloc_type, std::optional<nb::str> name)
        : GfxObject(ctx, true, std::move(name))
        , size(size)
    {
        VkResult vkr = gfx::CreateBuffer(&buffer, ctx->vk, size, {
            .usage = (VkBufferUsageFlags)usage_flags,
            .alloc = gfx::AllocPresets::Types[(size_t)alloc_type],
        });
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to create buffer");
        }

        VmaAllocationInfo2 alloc_info = {};
        vmaGetAllocationInfo2(ctx->vk.vma, buffer.allocation, &alloc_info);
        alloc = new AllocInfo(alloc_info);

        if (usage_flags & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
            device_address = gfx::GetBufferAddress(buffer.buffer, ctx->vk.device);
        }

        DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_BUFFER, buffer.buffer);
    }

    Buffer(nb::ref<Context> ctx, size_t size, std::optional<nb::str> name)
        : GfxObject(ctx, true, std::move(name))
        , size(size)
    { }

    ~Buffer() {
        destroy();
    }

    void destroy() {
        if(owned) {
            gfx::DestroyBuffer(&buffer, ctx->vk);
        }
    }

    static nb::ref<Buffer> from_data(nb::ref<Context> ctx, nb::object data, VkBufferUsageFlagBits usage_flags, gfx::AllocPresets::Type alloc_type, std::optional<nb::str> name) {
        Py_buffer view;
        if (PyObject_GetBuffer(data.ptr(), &view, PyBUF_SIMPLE) != 0) {
            throw nb::python_error();
        }

        if (!PyBuffer_IsContiguous(&view, 'C')) {
            PyBuffer_Release(&view);
            throw std::runtime_error("Data buffer must be contiguous");
        }

        std::unique_ptr<Buffer> self = std::make_unique<Buffer>(ctx, view.len, std::move(name));
        VkResult vkr = gfx::CreateBufferFromData(&self->buffer, ctx->vk, ArrayView<u8>((u8*)view.buf, view.len), {
            .usage = (VkBufferUsageFlags)usage_flags,
            .alloc = gfx::AllocPresets::Types[(size_t)alloc_type],
        });
        PyBuffer_Release(&view);

        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to create buffer");
        }

        VmaAllocationInfo2 alloc_info = {};
        vmaGetAllocationInfo2(ctx->vk.vma, self->buffer.allocation, &alloc_info);
        self->alloc = new AllocInfo(alloc_info);

        if (usage_flags & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
            self->device_address = gfx::GetBufferAddress(self->buffer.buffer, ctx->vk.device);
        }

        DEBUG_UTILS_OBJECT_NAME_WITH_NAME(VK_OBJECT_TYPE_BUFFER, self->buffer.buffer, self->name);
        return self.release();
    }

    static int bf_getbuffer(PyObject *exporter, Py_buffer *view, int flags) {
        Buffer* self = nb::inst_ptr<Buffer>(exporter);

        if (!self->buffer.map.data) {
            PyErr_SetString(PyExc_BufferError, "Buffer allocated in non-mappable memory");
            return -1;
        }

        // request-independent fields
        *view = {};
        view->obj = exporter;
        view->buf = self->buffer.map.data;
        view->len = self->buffer.map.length;
        view->itemsize = 1;
        view->ndim = 1;

        // readonly, format
        view->readonly = 0;
        if(flags & PyBUF_FORMAT) {
            view->format = (char*)"B";
        }

        // shape, strides, suboffsets
        if(flags & PyBUF_ND) {
            Py_ssize_t* shape = (Py_ssize_t*)PyMem_Malloc(sizeof(Py_ssize_t));
            *shape = self->buffer.map.length;
            view->shape = shape;
        }
        if(flags & PyBUF_STRIDES) {
            Py_ssize_t* strides = (Py_ssize_t*)PyMem_Malloc(sizeof(Py_ssize_t));
            *strides = 1;
            view->strides = strides;
        }
        view->suboffsets = NULL;
        view->internal = NULL;

        // Increse refcount of this object
        self->inc_ref();
        return 0;
    }

    static void bf_releasebuffer(PyObject *exporter, Py_buffer *view) {
        PyMem_Free(view->shape);
        PyMem_Free(view->strides);
    }

    gfx::Buffer buffer = {};
    size_t size = 0;
    std::optional<VkDeviceAddress> device_address;
    nb::ref<AllocInfo> alloc;
};

PyType_Slot buffer_slots[] = {
#if PY_VERSION_HEX >= 0x03090000
    { Py_bf_getbuffer, (void *) Buffer::bf_getbuffer },
    { Py_bf_releasebuffer, (void *) Buffer::bf_releasebuffer },
#endif
    { 0, nullptr }
};

struct Fence: GfxObject {
    Fence(nb::ref<Context> ctx, bool signaled, std::optional<nb::str> name)
        : GfxObject(ctx, true, std::move(name))
    {
        VkFenceCreateInfo fence_info = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        if (signaled) {
            fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        }
        VkResult vkr = vkCreateFence(ctx->vk.device, &fence_info, 0, &fence);
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to create fence");
        }

        DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_FENCE, fence);
    }

    bool is_signaled() {
        VkResult vkr = vkGetFenceStatus(ctx->vk.device, fence);
        if (vkr == VK_ERROR_DEVICE_LOST) {
            throw std::runtime_error("Device lost while checking fence status");
        }
        return vkr == VK_SUCCESS;
    }

    void wait() {
        nb::gil_scoped_release gil;
        vkWaitForFences(ctx->vk.device, 1, &fence, VK_TRUE, ~0U);
    }
    void reset() {
        vkResetFences(ctx->vk.device, 1, &fence);
    }

    void wait_and_reset() {
        wait();
        reset();
    }

    void destroy() {
        if (owned) {
            vkDestroyFence(ctx->vk.device, fence, 0);
            fence = VK_NULL_HANDLE;
        }
    }

    ~Fence() {
        destroy();
    }

    VkFence fence;
};

struct Semaphore: GfxObject {
    Semaphore(nb::ref<Context> ctx, std::optional<nb::str> name, bool external)
        : GfxObject(ctx, true, std::move(name))
    {
        VkResult vkr = gfx::CreateGPUSemaphore(ctx->vk.device, &semaphore, external);
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to create semaphore");
        }
        DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_SEMAPHORE, semaphore);
    }
    Semaphore(nb::ref<Context> ctx, std::optional<nb::str> name): Semaphore(ctx, std::move(name), false) { }

    Semaphore(): semaphore(VK_NULL_HANDLE) {}

    void destroy() {
        if (owned) {
            gfx::DestroyGPUSemaphore(ctx->vk.device, &semaphore);
        }
    }

    ~Semaphore() {
        destroy();
    }

    VkSemaphore semaphore;
};

struct TimelineSemaphore: Semaphore {
    TimelineSemaphore(nb::ref<Context> ctx, u64 initial_value, std::optional<nb::str> name, bool external)
    {
        this->ctx = ctx;
        this->owned = true;
        this->name = std::move(name);

        if (!(ctx->vk.device_features & gfx::DeviceFeatures::TIMELINE_SEMAPHORES)) {
            throw std::runtime_error("Device feature TIMELINE_SEMAPHORES must be set to use TimelineSemaphore");
        }

        VkResult vkr = gfx::CreateGPUTimelineSemaphore(ctx->vk.device, &this->semaphore, initial_value, external);
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to create semaphore");
        }

        DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_SEMAPHORE, semaphore);
    }

    void signal(u64 value) {
        VkSemaphoreSignalInfoKHR signal_info = { VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO_KHR };
        signal_info.semaphore = semaphore;
        signal_info.value = value;
        VkResult vkr = vkSignalSemaphoreKHR(ctx->vk.device, &signal_info);
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to get semaphore counter value");
        }
    }

    void wait(u64 value) {
        VkSemaphoreWaitInfoKHR wait_info = { VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO_KHR };
        wait_info.semaphoreCount = 1;
        wait_info.pSemaphores = &semaphore;
        wait_info.pValues = &value;

        VkResult vkr;
        {
            nb::gil_scoped_release gil;
            vkr = vkWaitSemaphoresKHR(ctx->vk.device, &wait_info, ~0ULL);
        }

        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to get semaphore counter value");
        }
    }

    u64 get_value() {
        u64 value;
        VkResult vkr = vkGetSemaphoreCounterValueKHR(ctx->vk.device, semaphore, &value);
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to get semaphore counter value");
        }
        return value;
    }

    TimelineSemaphore(nb::ref<Context> ctx, u64 initial_value, std::optional<nb::str> name): TimelineSemaphore(ctx, initial_value, std::move(name), false) { }
};

struct ExternalSemaphore: Semaphore {
    ExternalSemaphore(nb::ref<Context> ctx, std::optional<nb::str> name): Semaphore(ctx, std::move(name), true) {
        VkResult vkr = gfx::GetExternalHandleForSemaphore(&handle, ctx->vk, semaphore);
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to get handle for semaphore");
        }
    }

    ~ExternalSemaphore() {
        destroy();
    }

    void destroy() {
        if (owned) {
            gfx::CloseExternalHandle(&handle);
            Semaphore::destroy();
        }
    }

    gfx::ExternalHandle handle;
};

struct ExternalTimelineSemaphore: TimelineSemaphore {
    ExternalTimelineSemaphore(nb::ref<Context> ctx, u64 initial_value, std::optional<nb::str> name): TimelineSemaphore(ctx, initial_value, std::move(name), true) {
        VkResult vkr = gfx::GetExternalHandleForSemaphore(&handle, ctx->vk, semaphore);
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to get handle for semaphore");
        }
    }

    ~ExternalTimelineSemaphore() {
        destroy();
    }

    void destroy() {
        if (owned) {
            gfx::CloseExternalHandle(&handle);
            TimelineSemaphore::destroy();
        }
    }

    gfx::ExternalHandle handle;
};

struct ExternalBuffer: Buffer {
    ExternalBuffer(nb::ref<Context> ctx, usize size, VkBufferUsageFlagBits usage_flags, gfx::AllocPresets::Type alloc_type, std::optional<nb::str> name)
        : Buffer(ctx, size, std::move(name))
    {
        VkResult vkr;
        vkr = gfx::CreatePoolForBuffer(&pool, ctx->vk, {
            .usage = (VkBufferUsageFlags)usage_flags,
            .alloc = gfx::AllocPresets::Types[(size_t)alloc_type],
            .external = true,
        });
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pool");
        }

        vkr = gfx::CreateBuffer(&buffer, ctx->vk, size, {
            .usage = (VkBufferUsageFlags)usage_flags,
            .alloc = gfx::AllocPresets::Types[(size_t)alloc_type],
            .pool = pool.pool,
            .external = true,
        });
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to create buffer");
        }

        vkr = gfx::GetExternalHandleForBuffer(&handle, ctx->vk, buffer);
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to get external handle");
        }

        DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_BUFFER, buffer.buffer);
    }

    ~ExternalBuffer() {
        destroy();
    }

    void destroy() {
        if(owned) {
            gfx::CloseExternalHandle(&handle);
            Buffer::destroy();
            gfx::DestroyPool(&pool, ctx->vk);
        }
    }

    gfx::Pool pool;
    gfx::ExternalHandle handle = {};
};


bool has_any_write_access(VkAccessFlags2 flags) {
    return (flags & (VK_ACCESS_2_SHADER_WRITE_BIT
        | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
        | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
        | VK_ACCESS_2_TRANSFER_WRITE_BIT
        | VK_ACCESS_2_HOST_WRITE_BIT
        | VK_ACCESS_2_MEMORY_WRITE_BIT
        | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
        | VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR
        | VK_ACCESS_2_TRANSFORM_FEEDBACK_WRITE_BIT_EXT
        | VK_ACCESS_2_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT
        | VK_ACCESS_2_COMMAND_PREPROCESS_WRITE_BIT_NV
        | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR
        | VK_ACCESS_2_MICROMAP_WRITE_BIT_EXT
        | VK_ACCESS_2_OPTICAL_FLOW_WRITE_BIT_NV)) != 0;
}

enum class MemoryUsage {
    None,
    HostWrite,
    TransferSrc,
    TransferDst,
    VertexInput,
    VertexShaderUniform,
    GeometryShaderUniform,
    FragmentShaderUniform,
    ComputeShaderUniform,
    AnyShaderUniform,
    Image,
    ImageReadOnly,
    ImageWriteOnly,
    ShaderReadOnly,
    ColorAttachment,
    ColorAttachmentWriteOnly,
    DepthStencilAttachment,
    DepthStencilAttachmentReadOnly,
    DepthStencilAttachmentWriteOnly,
    Present,
    All,
    Count,
};

struct MemoryUsageState {
    VkPipelineStageFlagBits2 first_stage;
    VkPipelineStageFlagBits2 last_stage;
    VkAccessFlags2 access;
};

namespace MemoryUsagePresets {
    constexpr MemoryUsageState None {
        .first_stage = VK_PIPELINE_STAGE_2_NONE,
        .last_stage = VK_PIPELINE_STAGE_2_NONE,
        .access = 0,
    };
    constexpr MemoryUsageState HostWrite {
        .first_stage = VK_PIPELINE_STAGE_2_HOST_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_HOST_BIT,
        .access = VK_ACCESS_2_HOST_WRITE_BIT,
    };
    constexpr MemoryUsageState VertexInput {
        .first_stage = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT,
        .access = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,
    };
    constexpr MemoryUsageState TransferSrc = {
        .first_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .access = VK_ACCESS_2_TRANSFER_READ_BIT,
    };
    constexpr MemoryUsageState TransferDst {
        .first_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
    };
    constexpr MemoryUsageState VertexShaderUniform {
        .first_stage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
        .access = VK_ACCESS_2_UNIFORM_READ_BIT,
    };
    constexpr MemoryUsageState GeometryShaderUniform {
        .first_stage = VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT,
        .access = VK_ACCESS_2_UNIFORM_READ_BIT,
    };
    constexpr MemoryUsageState FragmentShaderUniform {
        .first_stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .access = VK_ACCESS_2_UNIFORM_READ_BIT,
    };
    constexpr MemoryUsageState ComputeShaderUniform {
        .first_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .access = VK_ACCESS_2_UNIFORM_READ_BIT,
    };
    constexpr MemoryUsageState AnyShaderUniform {
        .first_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .access = VK_ACCESS_2_UNIFORM_READ_BIT,
    };
    constexpr MemoryUsageState Image {
        .first_stage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
    };
    constexpr MemoryUsageState ImageReadOnly {
        .first_stage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
    };
    constexpr MemoryUsageState ImageWriteOnly {
        .first_stage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
    };
    constexpr MemoryUsageState ShaderReadOnly {
        .first_stage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .access = VK_ACCESS_2_SHADER_READ_BIT,
    };
    constexpr MemoryUsageState ColorAttachment = {
        .first_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .access = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
    };
    constexpr MemoryUsageState ColorAttachmentWriteOnly = {
        .first_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
    };
    constexpr MemoryUsageState DepthStencilAttachment = {
        .first_stage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        .access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
    };
    constexpr MemoryUsageState DepthStencilAttachmentReadOnly = {
        .first_stage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        .access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
    };
    constexpr MemoryUsageState DepthStencilAttachmentWriteOnly = {
        .first_stage = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        .access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
    };
    constexpr MemoryUsageState Present = {
        .first_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_NONE,
        .access = 0,
    };
    constexpr MemoryUsageState All = {
        .first_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .access = 0,
    };

    MemoryUsageState Types[] = {
        None,
        HostWrite,
        TransferSrc,
        TransferDst,
        VertexInput,
        VertexShaderUniform,
        GeometryShaderUniform,
        FragmentShaderUniform,
        ComputeShaderUniform,
        AnyShaderUniform,
        Image,
        ImageReadOnly,
        ImageWriteOnly,
        ShaderReadOnly,
        ColorAttachment,
        ColorAttachmentWriteOnly,
        DepthStencilAttachment,
        DepthStencilAttachmentReadOnly,
        DepthStencilAttachmentWriteOnly,
        Present,
        All,
    };
    static_assert(ArrayCount(Types) == (size_t)MemoryUsage::Count, "MemoryUsage count does not match length of Types array");
};

struct Image: GfxObject {
    Image(nb::ref<Context> ctx, u32 width, u32 height, VkFormat format, VkImageUsageFlagBits usage_flags, gfx::AllocPresets::Type alloc_type, int samples, std::optional<nb::str> name)
        : GfxObject(ctx, true, std::move(name))
        , width(width)
        , height(height)
        , format(format)
        , samples(samples)
    {
        VkResult vkr = gfx::CreateImage(&image, ctx->vk, {
            .width = width,
            .height = height,
            .format = format,
            .samples = (VkSampleCountFlagBits)samples,
            .usage = (VkBufferUsageFlags)usage_flags,
            .alloc = gfx::AllocPresets::Types[(size_t)alloc_type],
        });
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to create image");
        }

        VmaAllocationInfo2 alloc_info = {};
        vmaGetAllocationInfo2(ctx->vk.vma, image.allocation, &alloc_info);
        alloc = new AllocInfo(alloc_info);

        DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_IMAGE, image.image);
    }

    Image(nb::ref<Context> ctx, VkImage image, VkImageView view, u32 width, u32 height, VkFormat format, u32 samples)
        : GfxObject(ctx, false)
        , width(width)
        , height(height)
        , format(format)
        , samples(samples)
    {
        this->image.image = image;
        this->image.view = view;
        this->image.allocation = 0;
    }

    Image(nb::ref<Context> ctx, std::optional<nb::str> name)
        : GfxObject(ctx, true, std::move(name))
    { }

    ~Image() {
        destroy();
    }

    void destroy() {
        if(owned) {
            gfx::DestroyImage(&image, ctx->vk);
        }
    }

    static nb::ref<Image> from_data(nb::ref<Context> ctx, nb::object data, VkImageLayout layout,
        u32 width, u32 height, VkFormat format, VkImageUsageFlagBits usage_flags, gfx::AllocPresets::Type alloc_type, int samples, std::optional<nb::str> name)
    {
        Py_buffer view;
        if (PyObject_GetBuffer(data.ptr(), &view, PyBUF_SIMPLE) != 0) {
            throw nb::python_error();
        }

        if (!PyBuffer_IsContiguous(&view, 'C')) {
            PyBuffer_Release(&view);
            throw std::runtime_error("Data buffer must be contiguous");
        }

        std::unique_ptr<Image> self = std::make_unique<Image>(ctx, std::move(name));
        VkResult vkr = gfx::CreateAndUploadImage(&self->image, ctx->vk, ArrayView<u8>((u8*)view.buf, view.len), layout, {
            .width = width,
            .height = height,
            .format = format,
            .samples = (VkSampleCountFlagBits)samples,
            .usage = (VkBufferUsageFlags)usage_flags,
            .alloc = gfx::AllocPresets::Types[(size_t)alloc_type],
        });
        PyBuffer_Release(&view);

        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to create image");
        }

        VmaAllocationInfo2 alloc_info = {};
        vmaGetAllocationInfo2(ctx->vk.vma, self->image.allocation, &alloc_info);

        self->width = width;
        self->height = height;
        self->format = format;
        self->samples = samples;
        self->current_layout = layout;
        self->alloc = new AllocInfo(alloc_info);

        DEBUG_UTILS_OBJECT_NAME_WITH_NAME(VK_OBJECT_TYPE_IMAGE, self->image.image, self->name);

        return self.release();
    }

    gfx::Image image = {};
    u32 width;
    u32 height;
    VkFormat format;
    u32 samples;
    nb::ref<AllocInfo> alloc;
    VkImageLayout current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
};


struct Sampler: GfxObject {
    Sampler(
        nb::ref<Context> ctx,
        VkFilter min_filter,
        VkFilter mag_filter,
        VkSamplerMipmapMode mipmap_mode,
        float mip_lod_bias,
        float min_lod,
        float max_lod,
        VkSamplerAddressMode u,
        VkSamplerAddressMode v,
        VkSamplerAddressMode w,
        bool anisotroy_enabled,
        float max_anisotropy,
        bool compare_enable,
        VkCompareOp compare_op,
        std::optional<nb::str> name
    ) : GfxObject(ctx, true, std::move(name))
    {
        VkResult vkr = gfx::CreateSampler(&sampler, ctx->vk, gfx::SamplerDesc {
            .min_filter =        min_filter,
            .mag_filter =        mag_filter,
            .mipmap_mode =       mipmap_mode,
            .mip_lod_bias =      mip_lod_bias,
            .min_lod =           min_lod,
            .max_lod =           max_lod,
            .u =                 u,
            .v =                 v,
            .w =                 w,
            .anisotroy_enabled = anisotroy_enabled,
            .max_anisotropy =    max_anisotropy,
            .compare_enable =    compare_enable,
            .compare_op =        compare_op
        });

        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to create sampler");
        }

        DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_SAMPLER, sampler.sampler);
    }

    ~Sampler() {
        destroy();
    }

    void destroy() {
        if(owned) {
            gfx::DestroySampler(&sampler, ctx->vk);
        }
    }

    gfx::Sampler sampler;
};


struct Window;
struct GraphicsPipeline;
struct ComputePipeline;
struct DescriptorSet;
struct Buffer;

struct QueryPool: GfxObject {
    QueryPool(nb::ref<Context> ctx, VkQueryType type, u32 count, std::optional<nb::str> name)
        : GfxObject(ctx, true, std::move(name))
        , count(count)
        , type(type)
    {
        VkResult vkr = gfx::CreateQueryPool(&pool, ctx->vk, {
            .type = type,
            .count = count,
        });
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to create query pool");
        }
        DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_QUERY_POOL, pool);
    }

    ~QueryPool() {
        destroy();
    }

    void destroy() {
        if(owned) {
            gfx::DestroyQueryPool(&pool, ctx->vk);
        }
    }

    std::vector<u64> wait_results(u32 first, u32 count) {
        if ((u64)first + count > (u64)this->count) {
            nb::raise("Query range out of bounds");
        }

        std::vector<u64> data(count);
        VkResult vkr = vkGetQueryPoolResults(ctx->vk.device, pool, first, count, sizeof(u64) * count, data.data(), sizeof(u64),
                                             VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to get query pool results");
        }

        return data;
    }

    VkQueryPool pool;
    VkQueryType type;
    u32 count;
};

struct RenderingAttachment: nb::intrusive_base {
    nb::ref<Image> image;
    VkAttachmentLoadOp load_op;
    VkAttachmentStoreOp store_op;
    std::array<float, 4> clear;
    std::optional<nb::ref<Image>> resolve_image;
    VkResolveModeFlagBits resolve_mode;

    RenderingAttachment(nb::ref<Image> image, VkAttachmentLoadOp load_op, VkAttachmentStoreOp store_op, std::array<float, 4> clear, std::optional<nb::ref<Image>> resolve_image, VkResolveModeFlagBits resolve_mode)
        : image(image)
        , load_op(load_op)
        , store_op(store_op)
        , clear(clear)
        , resolve_image(resolve_image)
        , resolve_mode(resolve_mode)
    {}
};

struct DepthAttachment: nb::intrusive_base {
    nb::ref<Image> image;
    VkAttachmentLoadOp load_op;
    VkAttachmentStoreOp store_op;
    float clear;

    DepthAttachment(nb::ref<Image> image, VkAttachmentLoadOp load_op, VkAttachmentStoreOp store_op, float clear)
        : image(image)
        , load_op(load_op)
        , store_op(store_op)
        , clear(clear)
    {}
};

template<typename T>
void check_vector_of_ref_for_null(const std::vector<nb::ref<T>>& v, const char* error) {
    for (size_t i = 0; i < v.size(); i++) {
        if (!v[i]) {
            nb::raise("%s", error);
        }
    }
}

struct CommandBuffer: GfxObject {
    CommandBuffer(nb::ref<Context> ctx, std::optional<u32> queue_family_index, std::optional<nb::str> name)
        : GfxObject(ctx, true, std::move(name))
    {
        VkCommandPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pool_info.queueFamilyIndex = queue_family_index.value_or(ctx->vk.queue_family_index);

        VkResult vkr = vkCreateCommandPool(ctx->vk.device, &pool_info, 0, &pool);
        if (vkr != VK_SUCCESS) {
            nb::raise("Failed to create command pool");
        }

        VkCommandBufferAllocateInfo allocate_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        allocate_info.commandPool = pool;
        allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate_info.commandBufferCount = 1;

        vkr = vkAllocateCommandBuffers(ctx->vk.device, &allocate_info, &buffer);
        if (vkr != VK_SUCCESS) {
            nb::raise("Failed to create command buffer");
        }
        DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_COMMAND_BUFFER, buffer);
        DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_COMMAND_POOL, pool);
    }

    CommandBuffer(nb::ref<Context> ctx, VkCommandPool pool, VkCommandBuffer buffer)
        : GfxObject(ctx, false)
        , pool(pool)
        , buffer(buffer)
    {}

    void memory_barrier(MemoryUsage src_usage, MemoryUsage dst_usage) {
        assert((usize)src_usage < ArrayCount(MemoryUsagePresets::Types));
        assert((usize)dst_usage < ArrayCount(MemoryUsagePresets::Types));

        if (!(ctx->vk.device_features & gfx::DeviceFeatures::SYNCHRONIZATION_2)) {
            throw std::runtime_error("Device feature SYNCHRONIZATION_2 must be set to use memory_barrier");
        }

        MemoryUsageState src = MemoryUsagePresets::Types[(usize)src_usage];
        MemoryUsageState dst = MemoryUsagePresets::Types[(usize)dst_usage];

        // Rules:
        // - If one of the uses is a write: memory barrier needed
        if (has_any_write_access(src.access | dst.access)) {
            gfx::CmdMemoryBarrier(buffer, {
                .src_stage = src.last_stage,
                .src_access = src.access,
                .dst_stage = dst.first_stage,
                .dst_access = dst.access,
            });
        }
    }

    void buffer_barrier(nb::ref<Buffer> buf, MemoryUsage src_usage, MemoryUsage dst_usage, u32 src_queue_family_index, u32 dst_queue_family_index) {
        assert((usize)src_usage < ArrayCount(MemoryUsagePresets::Types));
        assert((usize)dst_usage < ArrayCount(MemoryUsagePresets::Types));

        if (!(ctx->vk.device_features & gfx::DeviceFeatures::SYNCHRONIZATION_2)) {
            throw std::runtime_error("Device feature SYNCHRONIZATION_2 must be set to use buffer_barrier");
        }

        // TODO: unify with above
        MemoryUsageState src = MemoryUsagePresets::Types[(usize)src_usage];
        MemoryUsageState dst = MemoryUsagePresets::Types[(usize)dst_usage];

        // Rules:
        // - If queue ownership transfer: always need an image barrier
        // - If one of the uses is a write: memory barrier needed
        if (src_queue_family_index != dst_queue_family_index) {
            gfx::CmdBufferBarrier(buffer, {
                .src_stage = src.last_stage,
                .src_access = src.access,
                .dst_stage = dst.first_stage,
                .dst_access = dst.access,
                .src_queue = src_queue_family_index,
                .dst_queue = dst_queue_family_index,
                .buffer = buf->buffer.buffer,
            });
        } else if (has_any_write_access(src.access | dst.access)) {
            gfx::CmdMemoryBarrier(buffer, {
                .src_stage = src.last_stage,
                .src_access = src.access,
                .dst_stage = dst.first_stage,
                .dst_access = dst.access,
            });
        }
    }

    void image_barrier(Image& image, VkImageLayout dst_layout, MemoryUsage src_usage, MemoryUsage dst_usage, u32 src_queue_family_index, u32 dst_queue_family_index, VkImageAspectFlagBits aspect_mask, bool undefined) {
        assert((usize)src_usage < ArrayCount(MemoryUsagePresets::Types));
        assert((usize)dst_usage < ArrayCount(MemoryUsagePresets::Types));

        if (!(ctx->vk.device_features & gfx::DeviceFeatures::SYNCHRONIZATION_2)) {
            throw std::runtime_error("Device feature SYNCHRONIZATION_2 must be set to use image_barrier");
        }

        // TODO: unify with above
        MemoryUsageState src_state = MemoryUsagePresets::Types[(usize)src_usage];
        MemoryUsageState dst_state = MemoryUsagePresets::Types[(usize)dst_usage];

        // Rules:
        // - If layout transition or queue ownership transfer: always need an image barrier
        // - If one of the uses is a write: memory barrier needed
        VkImageLayout src_layout = undefined ? VK_IMAGE_LAYOUT_UNDEFINED : image.current_layout;
        bool layout_transition = src_layout != dst_layout;
        bool queue_transfer = src_queue_family_index != dst_queue_family_index;
        if (layout_transition || queue_transfer) {
            gfx::CmdImageBarrier(buffer, {
                .src_stage   = src_state.last_stage,
                .src_access  = src_state.access,
                .dst_stage   = dst_state.first_stage,
                .dst_access  = dst_state.access,
                .old_layout  = src_layout,
                .new_layout  = dst_layout,
                .src_queue   = src_queue_family_index,
                .dst_queue   = dst_queue_family_index,
                .image       = image.image.image,
                .aspect_mask = (VkImageAspectFlags)aspect_mask,
            });
            image.current_layout = dst_layout;
        } else if (has_any_write_access(src_state.access | dst_state.access)) {
            gfx::CmdMemoryBarrier(buffer, {
                .src_stage  = src_state.last_stage,
                .src_access = src_state.access,
                .dst_stage  = dst_state.first_stage,
                .dst_access = dst_state.access,
            });
        }
    }

    void begin() {
        VkResult vkr = gfx::BeginCommands(pool, buffer, ctx->vk);
        if(vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to begin commands");
        }
    }

    void end() {
        VkResult vkr = gfx::EndCommands(buffer);
        if(vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to end commands");
        }
    }

    nb::ref<CommandBuffer> enter() {
        begin();
        return this;
    }

    void exit(nb::object, nb::object, nb::object) {
        end();
    }

    struct RenderingManager {
        RenderingManager(nb::ref<CommandBuffer> cmd, std::array<u32, 4> render_area, const std::vector<nb::ref<RenderingAttachment>>& color, std::optional<nb::ref<DepthAttachment>> depth)
            : cmd(cmd)
            , render_area(render_area)
            , color(std::move(color))
            , depth(depth)
        {}

        void enter() {
            cmd->begin_rendering(render_area, color, depth);
        }

        void exit(nb::object, nb::object, nb::object) {
            cmd->end_rendering();
        }

        nb::ref<CommandBuffer> cmd;
        std::array<u32, 4> render_area;
        std::vector<nb::ref<RenderingAttachment>> color;
        std::optional<nb::ref<DepthAttachment>> depth;
    };

    RenderingManager rendering(std::array<u32, 4> render_area, const std::vector<nb::ref<RenderingAttachment>>& color, std::optional<nb::ref<DepthAttachment>> depth) {
        return RenderingManager(this, render_area, std::move(color), depth);
    }

    void begin_rendering(std::array<u32, 4> render_area, const std::vector<nb::ref<RenderingAttachment>>& color, std::optional<nb::ref<DepthAttachment>> depth) {
        if (!(ctx->vk.device_features & gfx::DeviceFeatures::DYNAMIC_RENDERING)) {
            throw std::runtime_error("Device feature DYNAMIC_RENDERING must be set to use begin_rendering");
        }

        check_vector_of_ref_for_null(color, "elements of \"color\" must not be None");

        Array<gfx::RenderingAttachmentDesc> color_descs(color.size());
        for(usize i = 0; i < color_descs.length; i++) {
            const RenderingAttachment& attachment = *color[i];

            VkClearColorValue clear;
            clear.float32[0] = attachment.clear[0];
            clear.float32[1] = attachment.clear[1];
            clear.float32[2] = attachment.clear[2];
            clear.float32[3] = attachment.clear[3];

            color_descs[i] = {
                .view = attachment.image->image.view,
                .resolve_mode = attachment.resolve_mode,
                .resolve_image_view = attachment.resolve_image.has_value() ? attachment.resolve_image.value()->image.view : VK_NULL_HANDLE,
                .resolve_image_layout = attachment.resolve_image.has_value() ? attachment.resolve_image.value()->current_layout : VK_IMAGE_LAYOUT_UNDEFINED,
                .load_op = attachment.load_op,
                .store_op = attachment.store_op,
                .clear = clear,
            };
        }

        gfx::DepthAttachmentDesc depth_desc = {};
        if(depth.has_value()) {
            const DepthAttachment& depth_attachment = **depth;
            depth_desc.view = depth_attachment.image->image.view;
            depth_desc.load_op = depth_attachment.load_op;
            depth_desc.store_op= depth_attachment.store_op;
            depth_desc.clear = depth_attachment.clear;
        }

        gfx::CmdBeginRendering(buffer, {
            .color = Span(color_descs),
            .depth = depth_desc,
            .offset_x = render_area[0],
            .offset_y = render_area[1],
            .width = render_area[2],
            .height = render_area[3],
        });
    }

    void end_rendering() {
        if (!(ctx->vk.device_features & gfx::DeviceFeatures::DYNAMIC_RENDERING)) {
            throw std::runtime_error("Device feature DYNAMIC_RENDERING must be set to use begin_rendering");
        }

        gfx::CmdEndRendering(buffer);
    }

    void set_viewport(std::array<s32, 4> viewport) {
        VkViewport vp = {};
        vp.x = (float)viewport[0];
        vp.y = (float)viewport[1];
        vp.width = (float)viewport[2];
        vp.height = (float)viewport[3];
        vp.minDepth = 0.0f;
        vp.maxDepth = 1.0f;
        vkCmdSetViewport(buffer, 0, 1, &vp);
    }

    void set_scissors(std::array<s32, 4> scissors) {
        VkRect2D scissor = {};
        scissor.offset.x = scissors[0];
        scissor.offset.y = scissors[1];
        scissor.extent.width = scissors[2];
        scissor.extent.height = scissors[3];
        vkCmdSetScissor(buffer, 0, 1, &scissor);
    }

    void bind_pipeline(std::variant<nb::ref<GraphicsPipeline>, nb::ref<ComputePipeline>> pipeline);

    void bind_descriptor_sets(
        std::variant<nb::ref<GraphicsPipeline>, nb::ref<ComputePipeline>> pipeline,
        const std::vector<nb::ref<DescriptorSet>>& descriptor_sets,
        const std::vector<u32>& dynamic_offsets,
        u32 first_descriptor_set
    );

    void push_constants(
        std::variant<nb::ref<GraphicsPipeline>, nb::ref<ComputePipeline>> pipeline,
        const nb::bytes& push_constants,
        u32 offset
    );

    void bind_pipeline_common(
        VkPipelineBindPoint bind_point,
        VkPipeline pipeline,
        VkPipelineLayout layout,
        const std::vector<nb::ref<DescriptorSet>>& descriptor_sets,
        const std::vector<u32>& dynamic_offsets,
        u32 first_descriptor_set,
        const std::optional<nb::bytes>& push_constants,
        u32 push_constants_offset
    );

    void bind_compute_pipeline(
        const ComputePipeline& pipeline,
        const std::vector<nb::ref<DescriptorSet>>& descriptor_sets,
        const std::vector<u32>& dynamic_offsets,
        u32 first_descriptor_set,
        const std::optional<nb::bytes>& push_constants,
        u32 push_constants_offset
    );

    void bind_vertex_buffers(
        const std::vector<std::variant<nb::ref<Buffer>, std::tuple<nb::ref<Buffer>, VkDeviceSize>>>& vertex_buffers,
        u32 first_vertex_buffer_binding
    );

    void bind_index_buffer(
        std::optional<nb::ref<Buffer>> index_buffer,
        VkDeviceSize index_buffer_offset,
        VkIndexType index_type
    );

    void bind_graphics_pipeline(
        const GraphicsPipeline& pipeline,
        const std::vector<nb::ref<DescriptorSet>>& descriptor_sets,
        const std::vector<u32>& dynamic_offsets,
        u32 first_descriptor_set,
        const std::optional<nb::bytes>& push_constants,
        u32 push_constants_offset,
        const std::vector<std::variant<nb::ref<Buffer>, std::tuple<nb::ref<Buffer>, VkDeviceSize>>> vertex_buffers,
        u32 first_vertex_buffer_binding,
        std::optional<nb::ref<Buffer>> index_buffer,
        VkDeviceSize index_buffer_offset,
        VkIndexType index_type
    );

    void dispatch(
        u32 groups_x,
        u32 groups_y,
        u32 groups_z
    ) {
        vkCmdDispatch(buffer, groups_x, groups_y, groups_z);
    }

    void draw(
        u32 num_vertices,
        u32 num_instances,
        s32 first_vertex,
        u32 first_instance
    ) {
        vkCmdDraw(buffer, num_vertices, num_instances, first_vertex, first_instance);
    }

    void draw_indexed(
        u32 num_indices,
        u32 num_instances,
        u32 first_index,
        s32 vertex_offset,
        u32 first_instance
    ) {
        vkCmdDrawIndexed(buffer, num_indices, num_instances, first_index, vertex_offset, first_instance);
    }

    void copy_buffer(const Buffer& src, const Buffer& dst) {
        if (src.size != dst.size) {
            nb::raise("Buffer size mismatch. Src: %zu. Dst: %zu", src.size, dst.size);
        }

        gfx::CmdCopyBuffer(buffer, {
            .src = src.buffer.buffer,
            .dst = dst.buffer.buffer,
            .size = src.size,
        });
    }

    void copy_buffer_range(const Buffer& src, const Buffer& dst, VkDeviceSize size, VkDeviceSize src_offset, VkDeviceSize dst_offset) {
        if (src_offset + size > src.size) {
            nb::raise("Source buffer too small. Src size: %zu. Src offset: %zu. Size: %zu", src.size, src_offset, size);
        }
        if (dst_offset + size > dst.size) {
            nb::raise("Dest buffer too small. Dst size: %zu. Dst offset: %zu. Size: %zu", dst.size, dst_offset, size);
        }

        gfx::CmdCopyBuffer(buffer, {
            .src = src.buffer.buffer,
            .dst = dst.buffer.buffer,
            .src_offset = src_offset,
            .dst_offset = dst_offset,
            .size = size,
        });
    }

    void copy_image_to_buffer(Image& image, Buffer& buf, u64 buffer_offset_in_bytes) {
        // TODO: add error checking that image fits into buffer

        gfx::CmdCopyImageToBuffer(buffer, {
            .image = image.image.image,
            .image_layout = image.current_layout,
            .image_width = image.width,
            .image_height = image.height,
            .buffer = buf.buffer.buffer,
            .buffer_offset_in_bytes = buffer_offset_in_bytes,
        });
    }

    void copy_buffer_to_image(Buffer& buf, Image& image, u64 buffer_offset_in_bytes) {
        // TODO: add error checking that image fits into buffer

        gfx::CmdCopyBufferToImage(buffer, {
            .image = image.image.image,
            .image_layout = image.current_layout,
            .image_width = image.width,
            .image_height = image.height,
            .buffer = buf.buffer.buffer,
            .buffer_offset_in_bytes = buffer_offset_in_bytes,
        });
    }

    void copy_buffer_to_image_range(
        Buffer& buf,
        Image& image,
        u32 image_width,
        u32 image_height,
        u32 image_x,
        u32 image_y,
        u64 buffer_offset_in_bytes,
        u32 buffer_row_stride_in_texels
    ) {
        // TODO: add error checking that image fits into buffer
        // and that image range range fits into image

        gfx::CmdCopyBufferToImage(buffer, {
            .image = image.image.image,
            .image_layout = image.current_layout,
            .image_x = image_x,
            .image_y = image_y,
            .image_width = image_width,
            .image_height = image_height,
            .buffer = buf.buffer.buffer,
            .buffer_offset_in_bytes = buffer_offset_in_bytes,
            .buffer_row_stride_in_texels = buffer_row_stride_in_texels,
        });
    }


    void clear_color_image(Image& image, std::array<float, 4> color) {
        VkClearColorValue clear;
        clear.float32[0] = color[0];
        clear.float32[1] = color[1];
        clear.float32[2] = color[2];
        clear.float32[3] = color[3];

        VkImageSubresourceRange range = {};
        range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        range.layerCount = 1;
        range.levelCount = 1;
        vkCmdClearColorImage(buffer, image.image.image, image.current_layout, &clear, 1, &range);
    }

    void clear_depth_stencil_image(Image& image, std::optional<float> depth, std::optional<u32> stencil) {
        VkClearDepthStencilValue clear;
        clear.depth = depth.value_or(0.0f);
        clear.stencil = stencil.value_or(0);

        VkImageSubresourceRange range = {};
        range.aspectMask = (depth.has_value() ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | (stencil.has_value() ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
        range.layerCount = 1;
        range.levelCount = 1;
        vkCmdClearDepthStencilImage(buffer, image.image.image, image.current_layout, &clear, 1, &range);
    }


    void blit_image(Image& src, Image& dst, VkFilter filter, VkImageAspectFlagBits src_aspect, VkImageAspectFlagBits dst_aspect) {
        // TODO: add  error checking (check spec for what blits are valid)
        gfx::CmdBlitImage(buffer, {
            .src = src.image.image,
            .src_layout = src.current_layout,
            .src_width = src.width,
            .src_height = src.height,
            .src_aspect = (VkImageAspectFlags)src_aspect,
            .dst = dst.image.image,
            .dst_layout = dst.current_layout,
            .dst_width = dst.width,
            .dst_height = dst.height,
            .dst_aspect = (VkImageAspectFlags)dst_aspect,
            .filter = filter,
        });
    }

    void blit_image_range(Image& src, Image& dst, VkFilter filter,
        u32 src_width,
        u32 src_height,
        u32 src_x,
        u32 src_y,
        VkImageAspectFlagBits src_aspect,
        u32 dst_width,
        u32 dst_height,
        u32 dst_x,
        u32 dst_y,
        VkImageAspectFlagBits dst_aspect)
    {
        gfx::CmdBlitImage(buffer, {
            .src = src.image.image,
            .src_layout = src.current_layout,
            .src_x = src_x,
            .src_y = src_y,
            .src_width = src_width,
            .src_height = src_height,
            .src_aspect = (VkImageAspectFlags)src_aspect,
            .dst = dst.image.image,
            .dst_layout = dst.current_layout,
            .dst_x = dst_x,
            .dst_y = dst_y,
            .dst_width = dst_width,
            .dst_height = dst_height,
            .dst_aspect = (VkImageAspectFlags)dst_aspect,
            .filter = filter,
        });
    }

    void resolve_image(Image& src, Image& dst, VkImageAspectFlagBits src_aspect, VkImageAspectFlagBits dst_aspect)
    {
        gfx::CmdResolveImage(buffer, {
            .src = src.image.image,
            .src_layout = src.current_layout,
            .src_aspect = (VkImageAspectFlags)src_aspect,
            .dst = dst.image.image,
            .dst_layout = dst.current_layout,
            .dst_aspect = (VkImageAspectFlags)dst_aspect,
            .width = src.width,
            .height = src.height,
        });
    }

    void resolve_image_range(Image& src, Image& dst,
        u32 width,
        u32 height,
        u32 src_x,
        u32 src_y,
        VkImageAspectFlagBits src_aspect,
        u32 dst_x,
        u32 dst_y,
        VkImageAspectFlagBits dst_aspect)
    {
        gfx::CmdResolveImage(buffer, {
            .src = src.image.image,
            .src_layout = src.current_layout,
            .src_x = src_x,
            .src_y = src_y,
            .src_aspect = (VkImageAspectFlags)src_aspect,
            .dst = dst.image.image,
            .dst_layout = dst.current_layout,
            .dst_x = dst_x,
            .dst_y = dst_y,
            .dst_aspect = (VkImageAspectFlags)dst_aspect,
            .width = width,
            .height = height,
        });
    }


    void reset_query_pool(const QueryPool& pool) {
        vkCmdResetQueryPool(buffer, pool.pool, 0, pool.count);
    }

    void write_timestamp(const QueryPool& pool, u32 index, VkPipelineStageFlags2 stage) {
        vkCmdWriteTimestamp2KHR(buffer, stage, pool.pool, index);
    }

    void begin_label(nb::str name, std::optional<std::tuple<float, float, float, float>> color) {
        if (ctx->vk.debug_utils_enabled) {
            VkDebugUtilsLabelEXT label = { VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT };
            label.pLabelName = name.c_str();
            if (color.has_value()) {
                label.color[0] = std::get<0>(color.value());
                label.color[1] = std::get<1>(color.value());
                label.color[2] = std::get<2>(color.value());
                label.color[3] = std::get<3>(color.value());
            }
            vkCmdBeginDebugUtilsLabelEXT(buffer, &label);
        }
    }

    void end_label() {
        if (ctx->vk.debug_utils_enabled) {
            vkCmdEndDebugUtilsLabelEXT(buffer);
        }
    }

    void insert_label(nb::str name, std::optional<std::tuple<float, float, float, float>> color) {
        if (ctx->vk.debug_utils_enabled) {
            VkDebugUtilsLabelEXT label = { VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT };
            label.pLabelName = name.c_str();
            if (color.has_value()) {
                label.color[0] = std::get<0>(color.value());
                label.color[1] = std::get<1>(color.value());
                label.color[2] = std::get<2>(color.value());
                label.color[3] = std::get<3>(color.value());
            }
            vkCmdInsertDebugUtilsLabelEXT(buffer, &label);
        }
    }

    void set_line_width(float width) {
        vkCmdSetLineWidth(buffer, width);
    }

    void destroy() {
        if (owned) {
            vkFreeCommandBuffers(ctx->vk.device, pool, 1, &buffer);
            vkDestroyCommandPool(ctx->vk.device, pool, 0);
        }
    }

    ~CommandBuffer() {
        destroy();
    }

    VkCommandPool pool;
    VkCommandBuffer buffer;
};

struct Queue: GfxObject {
    Queue(nb::ref<Context> ctx, VkQueue queue)
        : GfxObject(ctx, false)
        , queue(queue)
    {}

    void begin_label(nb::str name, std::optional<std::tuple<float, float, float, float>> color) {
        if (ctx->vk.debug_utils_enabled) {
            VkDebugUtilsLabelEXT label = { VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT };
            label.pLabelName = name.c_str();
            if (color.has_value()) {
                label.color[0] = std::get<0>(color.value());
                label.color[1] = std::get<1>(color.value());
                label.color[2] = std::get<2>(color.value());
                label.color[3] = std::get<3>(color.value());
            }
            vkQueueBeginDebugUtilsLabelEXT(queue, &label);
        }
    }

    void end_label() {
        if (ctx->vk.debug_utils_enabled) {
            vkQueueEndDebugUtilsLabelEXT(queue);
        }
    }

    void insert_label(nb::str name, std::optional<std::tuple<float, float, float, float>> color) {
        if (ctx->vk.debug_utils_enabled) {
            VkDebugUtilsLabelEXT label = { VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT };
            label.pLabelName = name.c_str();
            if (color.has_value()) {
                label.color[0] = std::get<0>(color.value());
                label.color[1] = std::get<1>(color.value());
                label.color[2] = std::get<2>(color.value());
                label.color[3] = std::get<3>(color.value());
            }
            vkQueueInsertDebugUtilsLabelEXT(queue, &label);
        }
    }

    void submit(nb::ref<CommandBuffer> cmd,
        std::vector<std::tuple<nb::ref<Semaphore>, VkPipelineStageFlagBits>> wait_semaphores,
        std::vector<u64> wait_timeline_values,
        std::vector<nb::ref<Semaphore>> signal_semaphores,
        std::vector<u64> signal_timeline_values,
        std::optional<nb::ref<Fence>> fence) {

        check_vector_of_ref_for_null(signal_semaphores, "elements of \"signal_semaphores\" must not be None");
        for (size_t i = 0; i < wait_semaphores.size(); i++) {
            if (!std::get<0>(wait_semaphores[i])) {
                nb::raise("semaphores of \"wait_semaphores\" must not be None");
            }
        }

        Array<VkSemaphore> vk_wait_semaphores(wait_semaphores.size());
        Array<VkPipelineStageFlags> vk_wait_stages(wait_semaphores.size());
        for(usize i = 0; i < wait_semaphores.size(); i++) {
            vk_wait_semaphores[i] = std::get<0>(wait_semaphores[i])->semaphore;
            vk_wait_stages[i] = std::get<1>(wait_semaphores[i]);
        }
        Array<VkSemaphore> vk_signal_semaphores(signal_semaphores.size());
        for(usize i = 0; i < signal_semaphores.size(); i++) {
            vk_signal_semaphores[i] = signal_semaphores[i]->semaphore;
        }

        VkFence vk_fence = VK_NULL_HANDLE;
        if (fence.has_value()) {
            vk_fence = fence.value()->fence;
            vkResetFences(ctx->vk.device, 1, &vk_fence);
        }
        VkResult vkr = gfx::SubmitQueue(queue, {
            .cmd = { cmd->buffer },
            .wait_semaphores = Span(vk_wait_semaphores),
            .wait_stages = Span(vk_wait_stages),
            .wait_timeline_values = Span(ArrayView(wait_timeline_values.data(), wait_timeline_values.size())),
            .signal_semaphores = Span(vk_signal_semaphores),
            .signal_timeline_values = Span(ArrayView(signal_timeline_values.data(), signal_timeline_values.size())),
            .fence = vk_fence,
        });

        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit transfer queue commands");
        }
    }

    VkQueue queue;
};

struct Frame: nb::intrusive_base {
    Frame(nb::ref<Window> window, gfx::Frame& frame);

    gfx::Frame& frame;
    nb::ref<CommandBuffer> command_buffer;
    std::optional<nb::ref<CommandBuffer>> compute_command_buffer;
    std::optional<nb::ref<CommandBuffer>> transfer_command_buffer;
    nb::ref<Window> window;
    nb::ref<Image> image;
};

struct SwapchainOutOfDateError: std::exception {};

struct Window: nb::intrusive_base {
    struct FrameManager: nb::intrusive_base {
        FrameManager(nb::ref<Window> window,
                     std::vector<std::tuple<nb::ref<Semaphore>, VkPipelineStageFlagBits>> additional_wait_semaphores,
                     std::vector<u64> additional_wait_timeline_values,
                     std::vector<nb::ref<Semaphore>> additional_signal_semaphores,
                     std::vector<u64> additional_signal_timeline_values)
            : window(window)
            , additional_wait_semaphores(std::move(additional_wait_semaphores))
            , additional_wait_timeline_values(std::move(additional_wait_timeline_values))
            , additional_signal_semaphores(std::move(additional_signal_semaphores))
            , additional_signal_timeline_values(std::move(additional_signal_timeline_values))
        {
        }

        nb::ref<Frame> enter() {
            frame = window->begin_frame();
            return frame;
        }

        void exit(nb::object, nb::object, nb::object) {
            window->end_frame(*frame, additional_wait_semaphores, additional_wait_timeline_values, additional_signal_semaphores, additional_signal_timeline_values);
        }

        nb::ref<Frame> frame;
        nb::ref<Window> window;
        std::vector<std::tuple<nb::ref<Semaphore>, VkPipelineStageFlagBits>> additional_wait_semaphores;
        std::vector<u64> additional_wait_timeline_values;
        std::vector<nb::ref<Semaphore>> additional_signal_semaphores;
        std::vector<u64> additional_signal_timeline_values;
    };

    nb::ref<FrameManager> frame(
        std::vector<std::tuple<nb::ref<Semaphore>, VkPipelineStageFlagBits>> additional_wait_semaphores,
        std::vector<u64> additional_wait_timeline_values,
        std::vector<nb::ref<Semaphore>> additional_signal_semaphores,
        std::vector<u64> additional_signal_timeline_values)
    {
        return new FrameManager(this, std::move(additional_wait_semaphores), std::move(additional_wait_timeline_values), std::move(additional_signal_semaphores), std::move(additional_signal_timeline_values));
    }

    Window(nb::ref<Context> ctx, const std::string& name, u32 width, u32 height, std::optional<u32> x, std::optional<u32> y)
        : ctx(ctx)
    {
        if (CreateWindowWithSwapchain(&window, ctx->vk, name.c_str(), width, height, x.value_or(xpg::gfx::ANY_POSITION), y.value_or(xpg::gfx::ANY_POSITION)) != gfx::Result::SUCCESS) {
            throw std::runtime_error("Failed to create window");
        }
    }

    void set_callbacks(
        Function<void()> draw,
        Function<void(nb::tuple)> mouse_move_event,
        Function<void(nb::tuple, gfx::MouseButton, gfx::Action, gfx::Modifiers)> mouse_button_event,
        Function<void(nb::tuple, nb::tuple)> mouse_scroll_event,
        Function<void(gfx::Key, gfx::Action, gfx::Modifiers)> key_event
    )
    {
        this->draw               = std::move(draw);
        this->mouse_move_event   = std::move(mouse_move_event);
        this->mouse_button_event = std::move(mouse_button_event);
        this->mouse_scroll_event = std::move(mouse_scroll_event);
        this->key_event          = std::move(key_event);

        gfx::SetWindowCallbacks(&window, {
                .mouse_move_event = [this](glm::ivec2 p) {
                    nb::gil_scoped_acquire gil;
                    try {
                        if(this->mouse_move_event)
                            this->mouse_move_event(nb::make_tuple(p.x, p.y));
                    } catch (nb::python_error &e) {
                        e.restore();
                    }
                },
                .mouse_button_event = [this] (glm::ivec2 p, gfx::MouseButton b, gfx::Action a, gfx::Modifiers m) {
                    nb::gil_scoped_acquire gil;
                    try {
                        if(this->mouse_button_event)
                            this->mouse_button_event(nb::make_tuple(p.x, p.y), b, a, m);
                    } catch (nb::python_error &e) {
                        e.restore();
                    }
                },
                .mouse_scroll_event = [this] (glm::ivec2 p, glm::ivec2 s) {
                    nb::gil_scoped_acquire gil;
                    try {
                        if(this->mouse_scroll_event)
                            this->mouse_scroll_event(nb::make_tuple(p.x, p.y), nb::make_tuple(s.x, s.y));
                    } catch (nb::python_error &e) {
                        e.restore();
                    }
                },
                .key_event = [this] (gfx::Key k, gfx::Action a, gfx::Modifiers m) {
                    nb::gil_scoped_acquire gil;
                    try {
                        if(this->key_event)
                            this->key_event(k, a, m);
                    } catch (nb::python_error &e) {
                        e.restore();
                    }
                },
                .draw = [this] () {
                    nb::gil_scoped_acquire gil;
                    try {
                        if(this->draw)
                            this->draw();
                    } catch (nb::python_error &e) {
                        e.restore();
                    }
                }
        });
    }

    void reset_callbacks()
    {
        gfx::SetWindowCallbacks(&window, {});
    }

    gfx::SwapchainStatus update_swapchain()
    {
        gfx::SwapchainStatus status = gfx::UpdateSwapchain(&window, ctx->vk);
        if (status == gfx::SwapchainStatus::FAILED) {
            throw std::runtime_error("Failed to update swapchain");
        }
        return status;
    }

    nb::ref<Frame> begin_frame()
    {
        // TODO: make this throw if called multiple times in a row befor end
        gfx::Frame* frame;
        {
            nb::gil_scoped_release gil;
            frame = &gfx::WaitForFrame(&window, ctx->vk);
        }
        gfx::Result ok = gfx::AcquireImage(frame, &window, ctx->vk);

        if (ok == gfx::Result::SWAPCHAIN_OUT_OF_DATE) {
            throw SwapchainOutOfDateError();
        }

        if (ok != gfx::Result::SUCCESS) {
            throw std::runtime_error("Failed to acquire next image");
        }
        return new Frame(this, *frame);
    }

    void end_frame(Frame& frame,
        const std::vector<std::tuple<nb::ref<Semaphore>, VkPipelineStageFlagBits>>& additional_wait_semaphores,
        std::vector<u64>& additional_wait_timeline_values,
        const std::vector<nb::ref<Semaphore>>& additional_signal_semaphores,
        std::vector<u64>& additional_signal_timeline_values)
    {
        check_vector_of_ref_for_null(additional_signal_semaphores, "elements of \"additional_signal_semaphores\" must not be None");
        for (size_t i = 0; i < additional_wait_semaphores.size(); i++) {
            if (!std::get<0>(additional_wait_semaphores[i])) {
                nb::raise("semaphores of \"additional_wait_semaphores\" must not be None");
            }
        }

        // TODO: make this throw if not called after begin in the same frame
        VkResult vkr;
        if(additional_wait_semaphores.empty() && additional_signal_semaphores.empty()) {
            vkr = gfx::Submit(frame.frame, ctx->vk, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
            // vkr = gfx::Submit(frame.frame, ctx->vk, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
            // vkr = gfx::Submit(frame.frame, ctx->vk, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
        } else {
            Array<VkSemaphore> wait_semaphores(additional_wait_semaphores.size() + 1);
            Array<VkPipelineStageFlags> wait_stages(additional_wait_semaphores.size() + 1);
            for(usize i = 0; i < additional_wait_semaphores.size(); i++) {
                wait_semaphores[i] = std::get<0>(additional_wait_semaphores[i])->semaphore;
                wait_stages[i] = std::get<1>(additional_wait_semaphores[i]);
            }
            wait_semaphores[additional_wait_semaphores.size()] = frame.frame.acquire_semaphore;
            wait_stages[additional_wait_semaphores.size()] = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            additional_wait_timeline_values.push_back(0);

            Array<VkSemaphore> signal_semaphores(additional_signal_semaphores.size() + 1);
            for(usize i = 0; i < additional_signal_semaphores.size(); i++) {
                signal_semaphores[i] = additional_signal_semaphores[i]->semaphore;
            }
            signal_semaphores[additional_signal_semaphores.size()] = frame.frame.release_semaphore;
            additional_signal_timeline_values.push_back(0);

            vkr = gfx::SubmitQueue(ctx->vk.queue, {
                .cmd = { frame.frame.command_buffer },
                .wait_semaphores = Span(wait_semaphores),
                .wait_stages = Span(wait_stages),
                .wait_timeline_values = Span(ArrayView(additional_wait_timeline_values.data(), additional_wait_timeline_values.size())),
                .signal_semaphores = Span(signal_semaphores),
                .signal_timeline_values = Span(ArrayView(additional_signal_timeline_values.data(), additional_signal_timeline_values.size())),
                .fence = frame.frame.fence,
            });
        }
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit frame commands");
        }

        vkr = gfx::PresentFrame(&window, &frame.frame, ctx->vk);
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to present frame");
        }
    }

    void post_empty_event() {
        glfwPostEmptyEvent();
    }

    ~Window()
    {
        if (ctx) {
            gfx::WaitIdle(ctx->vk);
            gfx::DestroyWindowWithSwapchain(&window, ctx->vk);
        }
    }

    bool should_close() {
        return gfx::ShouldClose(window);
    }

    gfx::Modifiers get_modifiers_state() {
        return gfx::GetModifiersState(window);
    }

    nb::ref<Context> ctx;
    gfx::Window window;

    Function<void()> draw;
    Function<void(nb::tuple)> mouse_move_event;
    Function<void(nb::tuple, gfx::MouseButton, gfx::Action, gfx::Modifiers)> mouse_button_event;
    Function<void(nb::tuple, nb::tuple)> mouse_scroll_event;
    Function<void(gfx::Key, gfx::Action, gfx::Modifiers)> key_event;

    // Garbage collection:

    static int tp_traverse(PyObject *self, visitproc visit, void *arg) {
        // Retrieve a pointer to the C++ instance associated with 'self' (never fails)
        Window *w = nb::inst_ptr<Window>(self);

        // If w->value has an associated CPython object, return it.
        // If not, value.ptr() will equal NULL, which is also fine.
        nb::handle ctx                = nb::find(w->ctx.get());
        nb::handle draw               = nb::find(w->draw);
        nb::handle mouse_move_event   = nb::find(w->mouse_move_event);
        nb::handle mouse_button_event = nb::find(w->mouse_button_event);
        nb::handle mouse_scroll_event = nb::find(w->mouse_scroll_event);
        nb::handle key_event          = nb::find(w->key_event);

        // Inform the Python GC about the instance (if non-NULL)
        Py_VISIT(ctx.ptr());
        Py_VISIT(draw.ptr());
        Py_VISIT(mouse_move_event.ptr());
        Py_VISIT(mouse_button_event.ptr());
        Py_VISIT(mouse_scroll_event.ptr());
        Py_VISIT(key_event.ptr());

        return 0;
    }

    static int tp_clear(PyObject *self) {
        // Retrieve a pointer to the C++ instance associated with 'self' (never fails)
        Window *w = nb::inst_ptr<Window>(self);

        // Manually call destructor. The object will be left in a safe to destruct
        // state because the destructor will be called again.
        w->~Window();

        // Clear the cycle!
        w->ctx.reset();
        w->draw               = nullptr;
        w->mouse_move_event   = nullptr;
        w->mouse_button_event = nullptr;
        w->mouse_scroll_event = nullptr;
        w->key_event          = nullptr;

        return 0;
    }

};

// Slot data structure referencing the above two functions
static PyType_Slot window_tp_slots[] = {
    { Py_tp_traverse, (void*)Window::tp_traverse },
    { Py_tp_clear, (void*)Window::tp_clear },
    { 0, nullptr }
};

struct CommandsManager: nb::intrusive_base {
    CommandsManager(nb::ref<CommandBuffer> cmd,
                    VkQueue queue,
                    std::vector<std::tuple<nb::ref<Semaphore>, VkPipelineStageFlagBits>> wait_semaphores,
                    std::vector<u64> wait_timeline_values,
                    std::vector<nb::ref<Semaphore>> signal_semaphores,
                    std::vector<u64> signal_timeline_values,
                    VkFence fence,
                    bool wait_and_reset_fence)
        : cmd(cmd)
        , queue(queue)
        , wait_semaphores(std::move(wait_semaphores))
        , wait_timeline_values(std::move(wait_timeline_values))
        , signal_semaphores(std::move(signal_semaphores))
        , signal_timeline_values(std::move(signal_timeline_values))
        , fence(fence)
        , wait_and_reset_fence(wait_and_reset_fence)
    {
        check_vector_of_ref_for_null(signal_semaphores, "elements of \"signal_semaphores\" must not be None");
        for (size_t i = 0; i < wait_semaphores.size(); i++) {
            if (!std::get<0>(wait_semaphores[i])) {
                nb::raise("semaphores of \"wait_semaphores\" must not be None");
            }
        }
    }

    nb::ref<CommandBuffer> enter() {
        cmd->begin();
        return cmd;
    }

    void exit(nb::object, nb::object, nb::object) {
        cmd->end();

        Array<VkSemaphore> vk_wait_semaphores(wait_semaphores.size());
        Array<VkPipelineStageFlags> vk_wait_stages(wait_semaphores.size());
        for(usize i = 0; i < wait_semaphores.size(); i++) {
            vk_wait_semaphores[i] = std::get<0>(wait_semaphores[i])->semaphore;
            vk_wait_stages[i] = std::get<1>(wait_semaphores[i]);
        }
        Array<VkSemaphore> vk_signal_semaphores(signal_semaphores.size());
        for(usize i = 0; i < signal_semaphores.size(); i++) {
            vk_signal_semaphores[i] = signal_semaphores[i]->semaphore;
        }

        VkResult vkr = gfx::SubmitQueue(queue, {
            .cmd = { cmd->buffer },
            .wait_semaphores = Span(vk_wait_semaphores),
            .wait_stages = Span(vk_wait_stages),
            .wait_timeline_values = Span(ArrayView(wait_timeline_values.data(), wait_timeline_values.size())),
            .signal_semaphores = Span(vk_signal_semaphores),
            .signal_timeline_values = Span(ArrayView(signal_timeline_values.data(), signal_timeline_values.size())),
            .fence = fence,
        });

        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit transfer queue commands");
        }

        if (wait_and_reset_fence) {
            if (!fence) {
                throw std::runtime_error("Fence must not be None if wait_and_reset_fence is True");
            }

            {
                nb::gil_scoped_release gil;
                vkWaitForFences(cmd->ctx->vk.device, 1, &fence, VK_TRUE, ~0U);
            }
            vkResetFences(cmd->ctx->vk.device, 1, &fence);
        }
    }

    nb::ref<CommandBuffer> cmd;
    VkQueue queue;
    std::vector<std::tuple<nb::ref<Semaphore>, VkPipelineStageFlagBits>> wait_semaphores;
    std::vector<u64> wait_timeline_values;
    std::vector<nb::ref<Semaphore>> signal_semaphores;
    std::vector<u64> signal_timeline_values;
    VkFence fence;
    bool wait_and_reset_fence;
};

Frame::Frame(nb::ref<Window> window, gfx::Frame& frame)
    : window(window)
    , frame(frame)
{
    image = new Image(window->ctx, frame.current_image, frame.current_image_view, window->window.fb_width, window->window.fb_height, window->window.swapchain_format, 1);
    command_buffer = new CommandBuffer(window->ctx, frame.command_pool, frame.command_buffer);
    if (window->ctx->vk.compute_queue) {
        compute_command_buffer = new CommandBuffer(window->ctx, frame.compute_command_pool, frame.compute_command_buffer);
    }
    if (window->ctx->vk.copy_queue) {
        transfer_command_buffer = new CommandBuffer(window->ctx, frame.copy_command_pool, frame.copy_command_buffer);
    }
}

struct Gui: nb::intrusive_base {
    Gui(nb::ref<Window> window)
        : window(window)
    {
        gui::CreateImGuiImpl(&imgui_impl, window->window, window->ctx->vk, {});
    }

    struct GuiFrame {
        void enter()
        {
            gui::BeginFrame();
        }

        void exit(nb::object, nb::object, nb::object)
        {
            gui::EndFrame();
        }
    };

    GuiFrame frame() {
        return GuiFrame();
    }

    void begin_frame()
    {
        gui::BeginFrame();
    }

    void end_frame()
    {
        gui::EndFrame();
    }

    void render(CommandBuffer& command_buffer)
    {
        gui::Render(command_buffer.buffer);
    }

    void set_ini_filename(std::optional<nb::str> str) {
        ini_filename = std::move(str);
        ImGui::GetIO().IniFilename = ini_filename.has_value() ? ini_filename->c_str() : NULL;
    }

    ~Gui()
    {
        gfx::WaitIdle(window->ctx->vk);
        gui::DestroyImGuiImpl(&imgui_impl, window->ctx->vk);
    }

    nb::ref<Window> window;
    gui::ImGuiImpl imgui_impl;
    std::optional<nb::str> ini_filename;

    // Garbage collection:

    static int tp_traverse(PyObject *self, visitproc visit, void *arg) {
        // Retrieve a pointer to the C++ instance associated with 'self' (never fails)
        Gui *g = nb::inst_ptr<Gui>(self);

        // If w->value has an associated CPython object, return it.
        // If not, value.ptr() will equal NULL, which is also fine.
        nb::handle window = nb::find(g->window.get());

        // Inform the Python GC about the instance (if non-NULL)
        Py_VISIT(window.ptr());

        return 0;
    }

    static int tp_clear(PyObject *self) {
        // Retrieve a pointer to the C++ instance associated with 'self' (never fails)
        Gui *g = nb::inst_ptr<Gui>(self);

        // Clear the cycle!
        g->window.reset();

        return 0;
    }

};

// Slot data structure referencing the above two functions
static PyType_Slot gui_tp_slots[] = {
    { Py_tp_traverse, (void *) Gui::tp_traverse },
    { Py_tp_clear, (void *) Gui::tp_clear },
    { 0, nullptr }
};

struct AccelerationStructureMesh: gfx::AccelerationStructureMeshDesc {
    AccelerationStructureMesh(
        VkDeviceAddress vertices_address,
        u64 vertices_stride,
        u32 vertices_count,
        VkFormat vertices_format,
        VkDeviceAddress indices_address,
        VkIndexType indices_type,
        u32 primitive_count,
        std::array<float, 12> transform)
        : gfx::AccelerationStructureMeshDesc {
              .vertices_address = vertices_address,
              .vertices_stride = vertices_stride,
              .vertices_count = vertices_count,
              .vertices_format = vertices_format,
              .indices_address = indices_address,
              .indices_type = indices_type,
              .primitive_count = primitive_count,
              .transform = glm::mat4x3(
                transform[0], transform[ 1], transform[ 2],
                transform[3], transform[ 4], transform[ 5],
                transform[6], transform[ 7], transform[ 8],
                transform[9], transform[10], transform[11]
            ),
        }
    {
    }
};

static_assert(sizeof(AccelerationStructureMesh) == sizeof(gfx::AccelerationStructureMeshDesc));

struct AccelerationStructure: GfxObject {
    AccelerationStructure(nb::ref<Context> ctx, const std::vector<AccelerationStructureMesh>& meshes, bool prefer_fast_build, std::optional<nb::str> name)
        : GfxObject(ctx, true, std::move(name))
    {
        VkResult vkr = gfx::CreateAccelerationStructure(&as, ctx->vk, gfx::AccelerationStructureDesc {
            .meshes = ArrayView((gfx::AccelerationStructureMeshDesc*)meshes.data(), meshes.size()),
            .prefer_fast_build = prefer_fast_build,
        });
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to create acceleration structure");
        }
        DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR, as.tlas);
    }

    ~AccelerationStructure() {
        destroy();
    }

    void destroy() {
        if(owned) {
            gfx::DestroyAccelerationStructure(&as, ctx->vk);
        }
    }

    gfx::AccelerationStructure as;
};
struct DescriptorSetEntry: gfx::DescriptorSetEntryDesc {
    DescriptorSetEntry(u32 count, VkDescriptorType type)
        : gfx::DescriptorSetEntryDesc {
            .count = count,
            .type = type
        }
    {
    };
};
static_assert(sizeof(DescriptorSetEntry) == sizeof(gfx::DescriptorSetEntryDesc));

struct DescriptorSet: GfxObject {
    DescriptorSet(nb::ref<Context> ctx, const std::vector<DescriptorSetEntry>& entries, VkDescriptorBindingFlagBits flags, std::optional<nb::str> name)
        : GfxObject(ctx, true, std::move(name))
    {
        VkResult vkr = gfx::CreateDescriptorSet(&set, ctx->vk, {
            .entries = ArrayView((gfx::DescriptorSetEntryDesc*)entries.data(), entries.size()),
            .flags = (VkDescriptorBindingFlags)flags,
        });
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor set");
        }
        DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_DESCRIPTOR_SET, set.set);
        DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_DESCRIPTOR_POOL, set.pool);
        DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, set.layout);
    }

    void write_buffer(const Buffer& buffer, VkDescriptorType type, u32 binding, u32 element, VkDeviceSize offset, VkDeviceSize size) {
        gfx::WriteBufferDescriptor(set.set, ctx->vk, {
            .buffer = buffer.buffer.buffer,
            .type = type,
            .binding = binding,
            .element = element,
            .offset = offset,
            .size = size,
        });
    };

    void write_image(const Image& image, VkImageLayout layout, VkDescriptorType type, u32 binding, u32 element) {
        gfx::WriteImageDescriptor(set.set, ctx->vk, {
            .view = image.image.view,
            .layout = layout,
            .type = type,
            .binding = binding,
            .element = element,
        });
    };

    void write_sampler(const Sampler& sampler, u32 binding, u32 element) {
        gfx::WriteSamplerDescriptor(set.set, ctx->vk, {
            .sampler = sampler.sampler.sampler,
            .binding = binding,
            .element = element,
        });
    }

    void write_acceleration_structure(const AccelerationStructure& as, u32 binding, u32 element) {
        gfx::WriteAccelerationStructureDescriptor(set.set, ctx->vk, {
            .acceleration_structure = as.as.tlas,
            .binding = binding,
            .element = element,
        });
    }

    ~DescriptorSet()
    {
        destroy();
    }

    void destroy()
    {
        if (owned) {
            gfx::DestroyDescriptorSet(&set, ctx->vk);
        }
    }

    gfx::DescriptorSet set;
};

struct Shader: GfxObject {
    Shader(nb::ref<Context> ctx, const nb::bytes& code, std::optional<nb::str> name)
        : GfxObject(ctx, true, std::move(name))
    {
        VkResult vkr = gfx::CreateShader(&shader, ctx->vk, ArrayView<u8>((u8*)code.data(), code.size()));
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to create shader");
        }
        DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_SHADER_MODULE, shader.shader);
    }

    ~Shader() {
        destroy();
    }

    void destroy() {
        if (owned) {
            gfx::DestroyShader(&shader, ctx->vk);
        }
    }

    gfx::Shader shader;
};

struct PipelineStage: nb::intrusive_base {
    PipelineStage(nb::ref<Shader> shader, VkShaderStageFlagBits stage, std::string entry)
        : shader(shader)
        , stage(stage)
        , entry(std::move(entry)) {
    };

    nb::ref<Shader> shader;
    VkShaderStageFlagBits stage;
    std::string entry;
};

struct VertexBinding: gfx::VertexBindingDesc {
    VertexBinding(u32 binding, u32 stride, VkVertexInputRate input_rate)
        : gfx::VertexBindingDesc {
            .binding = binding,
            .stride = stride,
            .input_rate = input_rate,
        } {}
};
static_assert(sizeof(VertexBinding) == sizeof(gfx::VertexBindingDesc));

struct VertexAttribute: gfx::VertexAttributeDesc {
    VertexAttribute(u32 location, u32 binding, VkFormat format, u32 offset)
        : gfx::VertexAttributeDesc {
              .location = location,
              .binding = binding,
              .format = format,
              .offset = offset,
          }
    {
    }
};
static_assert(sizeof(VertexAttribute) == sizeof(gfx::VertexAttributeDesc));

struct InputAssembly: gfx::InputAssemblyDesc {
    InputAssembly(VkPrimitiveTopology primitive_topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, bool primitive_restart_enable = false)
        : gfx::InputAssemblyDesc {
              .primitive_topology = (VkPrimitiveTopology)primitive_topology,
              .primitive_restart_enable = primitive_restart_enable,
          }
    {
    }
};
static_assert(sizeof(InputAssembly) == sizeof(gfx::InputAssemblyDesc));

struct Rasterization: gfx::RasterizationDesc {
    Rasterization(
        VkPolygonMode polygon_mode = VK_POLYGON_MODE_FILL,
        VkCullModeFlagBits cull_mode = VK_CULL_MODE_NONE,
        VkFrontFace front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        bool depth_bias_enable = false,
        bool depth_clamp_enable = false,
        bool dynamic_line_width = false,
        float line_width = 1.0f
    )
        : gfx::RasterizationDesc{
              .polygon_mode = polygon_mode,
              .cull_mode = (VkCullModeFlags)cull_mode,
              .front_face = front_face,
              .depth_bias_enable = depth_bias_enable,
              .depth_clamp_enable = depth_clamp_enable,
              .dynamic_line_width = dynamic_line_width,
              .line_width = line_width,
          }
    {
    }
};
static_assert(sizeof(Rasterization) == sizeof(gfx::RasterizationDesc));

struct Depth: gfx::DepthDesc {
    Depth(VkFormat format, bool test = false, bool write = false, VkCompareOp op = VK_COMPARE_OP_LESS)
        : gfx::DepthDesc {
            .test = test,
            .write = write,
            .op = op,
            .format = format,
        }
    {
    }
};
static_assert(sizeof(Depth) == sizeof(gfx::DepthDesc));

struct Attachment: gfx::AttachmentDesc {
    Attachment(
        VkFormat format,
        bool blend_enable = false,
        VkBlendFactor src_color_blend_factor = VK_BLEND_FACTOR_ZERO,
        VkBlendFactor dst_color_blend_factor = VK_BLEND_FACTOR_ZERO,
        VkBlendOp color_blend_op = VK_BLEND_OP_ADD,
        VkBlendFactor src_alpha_blend_factor = VK_BLEND_FACTOR_ZERO,
        VkBlendFactor dst_alpha_blend_factor = VK_BLEND_FACTOR_ZERO,
        VkBlendOp alpha_blend_op = VK_BLEND_OP_ADD,
        VkColorComponentFlags color_write_mask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
    )
        : gfx::AttachmentDesc {
            format = format,
            blend_enable = blend_enable,
            src_color_blend_factor = src_color_blend_factor,
            dst_color_blend_factor = dst_color_blend_factor,
            color_blend_op = color_blend_op,
            src_alpha_blend_factor = src_alpha_blend_factor,
            dst_alpha_blend_factor = dst_alpha_blend_factor,
            alpha_blend_op = alpha_blend_op,
            color_write_mask = color_write_mask,
        }
    {
    }
};
static_assert(sizeof(Attachment) == sizeof(gfx::AttachmentDesc));

struct PushConstantsRange: gfx::PushConstantsRangeDesc {
    PushConstantsRange(u32 size, u32 offset, VkShaderStageFlagBits flags)
        : gfx::PushConstantsRangeDesc {
            .flags = (VkShaderStageFlags)flags,
            .offset = offset,
            .size = size,
        }
    {
    }
};
static_assert(sizeof(PushConstantsRange) == sizeof(gfx::PushConstantsRangeDesc));

struct ComputePipeline: GfxObject {
    ComputePipeline(nb::ref<Context> ctx,
        nb::ref<Shader> shader,
        nb::str entry,
        const std::vector<PushConstantsRange>& push_constant_ranges,
        const std::vector<nb::ref<DescriptorSet>>& descriptor_sets,
        std::optional<nb::str> name
        )
        : GfxObject(ctx, true, std::move(name))
    {
        check_vector_of_ref_for_null(descriptor_sets, "elements of \"descriptor_sets\" must not be None");

        Array<VkDescriptorSetLayout> d(descriptor_sets.size());
        for(usize i = 0; i < d.length; i++) {
            d[i] = descriptor_sets[i]->set.layout;
        }

        VkResult vkr = gfx::CreateComputePipeline(&pipeline, ctx->vk, {
            .shader = shader->shader,
            .entry = entry.c_str(),
            .push_constants = ArrayView((gfx::PushConstantsRangeDesc*)push_constant_ranges.data(), push_constant_ranges.size()),
            .descriptor_sets = ArrayView(d),
        });

        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to create graphics pipeline");
        }
        DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_PIPELINE, pipeline.pipeline);
        DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_PIPELINE_LAYOUT, pipeline.layout);
    }

    ~ComputePipeline() {
        destroy();
    }

    void destroy() {
        if (owned) {
            gfx::DestroyComputePipeline(&pipeline, ctx->vk);
        }
    }

    gfx::ComputePipeline pipeline;
};

struct GraphicsPipeline: GfxObject {
    GraphicsPipeline(nb::ref<Context> ctx,
        const std::vector<nb::ref<PipelineStage>>& stages,
        const std::vector<VertexBinding>& vertex_bindings,
        const std::vector<VertexAttribute>& vertex_attributes,
        InputAssembly input_assembly,
        Rasterization rasterization,
        const std::vector<PushConstantsRange>& push_constant_ranges,
        const std::vector<nb::ref<DescriptorSet>>& descriptor_sets,
        u32 samples,
        const std::vector<Attachment>& attachments,
        Depth depth,
        std::optional<nb::str> name
        )
        : GfxObject(ctx, true, std::move(name))
    {
        check_vector_of_ref_for_null(stages, "elements of \"stages\" must not be None");
        check_vector_of_ref_for_null(descriptor_sets, "elements of \"descriptor_sets\" must not be None");

        Array<gfx::PipelineStageDesc> s(stages.size());
        for(usize i = 0; i < s.length; i++) {
            s[i].shader = stages[i]->shader->shader;
            s[i].stage = (VkShaderStageFlagBits)stages[i]->stage;
            s[i].entry = stages[i]->entry.c_str();
        }

        Array<VkDescriptorSetLayout> d(descriptor_sets.size());
        for(usize i = 0; i < d.length; i++) {
            d[i] = descriptor_sets[i]->set.layout;
        }

        VkResult vkr = gfx::CreateGraphicsPipeline(&pipeline, ctx->vk, {
            .stages = ArrayView(s),
            .vertex_bindings = ArrayView((gfx::VertexBindingDesc*)vertex_bindings.data(), vertex_bindings.size()),
            .vertex_attributes = ArrayView((gfx::VertexAttributeDesc*)vertex_attributes.data(), vertex_attributes.size()),
            .input_assembly = input_assembly,
            .rasterization = rasterization,
            .samples = (VkSampleCountFlagBits)samples,
            .depth = depth,
            .push_constants = ArrayView((gfx::PushConstantsRangeDesc*)push_constant_ranges.data(), push_constant_ranges.size()),
            .descriptor_sets = ArrayView(d),
            .attachments = ArrayView((gfx::AttachmentDesc*)attachments.data(), attachments.size()),
        });

        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to create graphics pipeline");
        }
        DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_PIPELINE, pipeline.pipeline);
        DEBUG_UTILS_OBJECT_NAME(VK_OBJECT_TYPE_PIPELINE_LAYOUT, pipeline.layout);
    }

    ~GraphicsPipeline() {
        destroy();
    }

    void destroy() {
        if (owned) {
            gfx::DestroyGraphicsPipeline(&pipeline, ctx->vk);
        }
    }

    gfx::GraphicsPipeline pipeline;
};

void CommandBuffer::bind_pipeline(std::variant<nb::ref<GraphicsPipeline>, nb::ref<ComputePipeline>> pipeline) {
    if (std::holds_alternative<nb::ref<GraphicsPipeline>>(pipeline)) {
        nb::ref<GraphicsPipeline> graphics_pipeline = std::get<nb::ref<GraphicsPipeline>>(pipeline);
        vkCmdBindPipeline(buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline->pipeline.pipeline);
    } else {
        nb::ref<ComputePipeline> compute_pipeline = std::get<nb::ref<ComputePipeline>>(pipeline);
        vkCmdBindPipeline(buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline->pipeline.pipeline);
    }
}

void CommandBuffer::bind_descriptor_sets(
    std::variant<nb::ref<GraphicsPipeline>, nb::ref<ComputePipeline>> pipeline,
    const std::vector<nb::ref<DescriptorSet>>& descriptor_sets,
    const std::vector<u32>& dynamic_offsets,
    u32 first_descriptor_set)
{
    check_vector_of_ref_for_null(descriptor_sets, "elements of \"descriptor_sets\" must not be None");

    VkPipelineLayout layout;
    VkPipelineBindPoint bind_point;
    if (std::holds_alternative<nb::ref<GraphicsPipeline>>(pipeline)) {
        nb::ref<GraphicsPipeline> graphics_pipeline = std::get<nb::ref<GraphicsPipeline>>(pipeline);
        layout = graphics_pipeline->pipeline.layout;
        bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
    } else {
        nb::ref<ComputePipeline> compute_pipeline = std::get<nb::ref<ComputePipeline>>(pipeline);
        layout = compute_pipeline->pipeline.layout;
        bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
    }

    // Descriptor sets
    if(descriptor_sets.size() > 0) {
        Array<VkDescriptorSet> sets(descriptor_sets.size());
        for(usize i = 0; i < sets.length; i++) {
            sets[i] = descriptor_sets[i]->set.set;
        }
        vkCmdBindDescriptorSets(buffer, bind_point, layout, first_descriptor_set, sets.length, sets.data, (u32)dynamic_offsets.size(), dynamic_offsets.data());
    }
}

void CommandBuffer::push_constants(
    std::variant<nb::ref<GraphicsPipeline>, nb::ref<ComputePipeline>> pipeline,
    const nb::bytes& push_constants,
    u32 offset)
{
    VkPipelineLayout layout;
    if (std::holds_alternative<nb::ref<GraphicsPipeline>>(pipeline)) {
        nb::ref<GraphicsPipeline> graphics_pipeline = std::get<nb::ref<GraphicsPipeline>>(pipeline);
        layout = graphics_pipeline->pipeline.layout;
    } else {
        nb::ref<ComputePipeline> compute_pipeline = std::get<nb::ref<ComputePipeline>>(pipeline);
        layout = compute_pipeline->pipeline.layout;
    }

    vkCmdPushConstants(buffer, layout, VK_SHADER_STAGE_ALL, offset, push_constants.size(), push_constants.data());
}

void CommandBuffer::bind_pipeline_common(
    VkPipelineBindPoint bind_point,
    VkPipeline pipeline,
    VkPipelineLayout layout,
    const std::vector<nb::ref<DescriptorSet>>& descriptor_sets,
    const std::vector<u32>& dynamic_offsets,
    u32 first_descriptor_set,
    const std::optional<nb::bytes>& push_constants,
    u32 push_constants_offset)
{
    // Pipeline
    vkCmdBindPipeline(buffer, bind_point, pipeline);

    // Descriptor sets
    if(descriptor_sets.size() > 0) {
        Array<VkDescriptorSet> sets(descriptor_sets.size());
        for(usize i = 0; i < sets.length; i++) {
            sets[i] = descriptor_sets[i]->set.set;
        }
        vkCmdBindDescriptorSets(buffer, bind_point, layout, first_descriptor_set, sets.length, sets.data, (u32)dynamic_offsets.size(), dynamic_offsets.data());
    }

    // Push constants
    if(push_constants.has_value()) {
        vkCmdPushConstants(buffer, layout, VK_SHADER_STAGE_ALL, push_constants_offset, push_constants->size(), push_constants->data());
    }
}

void CommandBuffer::bind_compute_pipeline(
    const ComputePipeline& pipeline,
    const std::vector<nb::ref<DescriptorSet>>& descriptor_sets,
    const std::vector<u32>& dynamic_offsets,
    u32 first_descriptor_set,
    const std::optional<nb::bytes>& push_constants,
    u32 push_constants_offset)
{
    check_vector_of_ref_for_null(descriptor_sets, "elements of \"descriptor_sets\" must not be None");
    bind_pipeline_common(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline.pipeline, pipeline.pipeline.layout, descriptor_sets, dynamic_offsets, first_descriptor_set, push_constants, push_constants_offset);
}

void CommandBuffer::bind_vertex_buffers(
    const std::vector<std::variant<nb::ref<Buffer>, std::tuple<nb::ref<Buffer>, VkDeviceSize>>>& vertex_buffers,
    u32 first_vertex_buffer_binding
)
{
    if(vertex_buffers.size() > 0) {
        Array<VkDeviceSize> offsets(vertex_buffers.size());
        Array<VkBuffer> buffers(vertex_buffers.size());
        for(usize i = 0; i < vertex_buffers.size(); i++) {
            if (std::holds_alternative<nb::ref<Buffer>>(vertex_buffers[i])) {
                nb::ref<Buffer> ref = std::get<nb::ref<Buffer>>(vertex_buffers[i]);
                if (!ref) {
                    nb::raise("elements of vertex_buffers must not be None");
                }
                offsets[i] = 0;
                buffers[i] = ref->buffer.buffer;
            } else {
                std::tuple<nb::ref<Buffer>, VkDeviceSize> tuple = std::get<std::tuple<nb::ref<Buffer>, VkDeviceSize>>(vertex_buffers[i]);
                nb::ref<Buffer> ref = std::get<0>(tuple);
                if (!ref) {
                    nb::raise("buffer elements of vertex_buffers must not be None");
                }
                offsets[i] = std::get<1>(tuple);
                buffers[i] = ref->buffer.buffer;
            }
        }
        vkCmdBindVertexBuffers(buffer, first_vertex_buffer_binding, buffers.length, buffers.data, offsets.data);
    }
}

void CommandBuffer::bind_index_buffer(
    std::optional<nb::ref<Buffer>> index_buffer,
    VkDeviceSize index_buffer_offset,
    VkIndexType index_type)
{
    vkCmdBindIndexBuffer(buffer, index_buffer.value()->buffer.buffer, index_buffer_offset, index_type);
}

void CommandBuffer::bind_graphics_pipeline(
    const GraphicsPipeline& pipeline,
    const std::vector<nb::ref<DescriptorSet>>& descriptor_sets,
    const std::vector<u32>& dynamic_offsets,
    u32 first_descriptor_set,
    const std::optional<nb::bytes>& push_constants,
    u32 push_constants_offset,
    const std::vector<std::variant<nb::ref<Buffer>, std::tuple<nb::ref<Buffer>, VkDeviceSize>>> vertex_buffers,
    u32 first_vertex_buffer_binding,
    std::optional<nb::ref<Buffer>> index_buffer,
    VkDeviceSize index_buffer_offset,
    VkIndexType index_type)
{
    check_vector_of_ref_for_null(descriptor_sets, "elements of \"descriptor_sets\" must not be None");
    bind_pipeline_common(VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline.pipeline, pipeline.pipeline.layout, descriptor_sets, dynamic_offsets, first_descriptor_set, push_constants, push_constants_offset);
    bind_vertex_buffers(vertex_buffers, first_vertex_buffer_binding);

    // Index buffers
    if(index_buffer.has_value()) {
        vkCmdBindIndexBuffer(buffer, index_buffer.value()->buffer.buffer, index_buffer_offset, index_type);
    }
}

#ifdef _WIN32
BOOL WINAPI ctrlc_handler(DWORD) {
    glfwPostEmptyEvent();
    return FALSE;
}
#endif

u64 mul_div(u64 ticks, u64 mul, u64 div)
{
    // (ticks * mul) / div == (ticks / div) * mul + (ticks % div) * mul / div
    u64 intpart, remaining;
    intpart = ticks / div;
    ticks %= div;
    remaining = ticks * mul;
    remaining /= div;
    return intpart * mul + remaining;
}

void gfx_create_bindings(nb::module_& m)
{
#ifdef _WIN32
    SetConsoleCtrlHandler(ctrlc_handler, TRUE);
#endif
    nb::exception<SwapchainOutOfDateError>(m, "SwapchainOutOfDateError");

    nb::enum_<VkMemoryHeapFlagBits>(m, "MemoryHeapFlags", nb::is_flag())
        .value("VK_MEMORY_HEAP_DEVICE_LOCAL"  , VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
        .value("VK_MEMORY_HEAP_MULTI_INSTANCE", VK_MEMORY_HEAP_MULTI_INSTANCE_BIT)
    ;

    nb::enum_<VkMemoryPropertyFlagBits>(m, "MemoryPropertyFlags", nb::is_flag())
        .value("DEVICE_LOCAL"     , VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        .value("HOST_VISIBLE"     , VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        .value("HOST_COHERENT"    , VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        .value("HOST_CACHED"      , VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
        .value("LAZILY_ALLOCATED" , VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT)
        .value("PROTECTED"        , VK_MEMORY_PROPERTY_PROTECTED_BIT)
        .value("DEVICE_COHERENT"  , VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD)
        .value("DEVICE_UNCACHED"  , VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD)
        .value("RDMA_CAPABLE"     , VK_MEMORY_PROPERTY_RDMA_CAPABLE_BIT_NV)
    ;

    nb::class_<MemoryHeap>(m, "MemoryHeap",
        nb::intrusive_ptr<MemoryHeap>([](MemoryHeap *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def_ro("size", &MemoryHeap::size)
        .def_ro("flags", &MemoryHeap::flags)
        .def("__repr__", [](MemoryHeap& h) {
            return nb::str("MemoryHeap(size={}, flags={})").format(h.size, h.flags);
        })
    ;

    nb::class_<MemoryType>(m, "MemoryType",
        nb::intrusive_ptr<MemoryType>([](MemoryType *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def_ro("heap_index", &MemoryType::heap_index)
        .def_ro("property_flags", &MemoryType::property_flags)
        .def("__repr__", [](MemoryType& t) {
            return nb::str("MemoryType(heap_index={}, property_flags={})").format(t.heap_index, t.property_flags);
        })
    ;

    nb::class_<MemoryProperties>(m, "MemoryProperties",
        nb::intrusive_ptr<MemoryProperties>([](MemoryProperties *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def_ro("memory_heaps", &MemoryProperties::memory_heaps)
        .def_ro("memory_types", &MemoryProperties::memory_types)
        .def("__repr__", [](MemoryProperties& memory_properties) {
            return nb::str("MemoryProperties(memory_heaps={}, memory_types={})").format(memory_properties.memory_heaps, memory_properties.memory_types);
        })
    ;

    nb::class_<HeapStatistics>(m, "HeapStatistics",
        nb::intrusive_ptr<HeapStatistics>([](HeapStatistics *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def_ro("block_count",      &HeapStatistics::block_count)
        .def_ro("allocation_count", &HeapStatistics::allocation_count)
        .def_ro("block_bytes",      &HeapStatistics::block_bytes)
        .def_ro("allocation_bytes", &HeapStatistics::allocation_bytes)
        .def_ro("usage",            &HeapStatistics::usage)
        .def_ro("budget",           &HeapStatistics::budget)
        .def("__repr__", [](HeapStatistics& stats) {
            return nb::str("HeapStatistics(block_count={}, allocation_count={}, block_bytes={}, allocation_bytes={}, usage={}, budget={})").format(stats.block_count, stats.allocation_count, stats.block_bytes, stats.allocation_bytes, stats.usage, stats.budget);
        })
    ;

    nb::enum_<gfx::DeviceFeatures::Flags>(m, "DeviceFeatures", nb::is_flag(), nb::is_arithmetic())
        .value("NONE",                  gfx::DeviceFeatures::NONE)
        .value("DYNAMIC_RENDERING",     gfx::DeviceFeatures::DYNAMIC_RENDERING)
        .value("SYNCHRONIZATION_2",     gfx::DeviceFeatures::SYNCHRONIZATION_2)
        .value("DESCRIPTOR_INDEXING",   gfx::DeviceFeatures::DESCRIPTOR_INDEXING)
        .value("SCALAR_BLOCK_LAYOUT",   gfx::DeviceFeatures::SCALAR_BLOCK_LAYOUT)
        .value("RAY_QUERY",             gfx::DeviceFeatures::RAY_QUERY)
        .value("RAY_PIPELINE",          gfx::DeviceFeatures::RAY_TRACING_PIPELINE)
        .value("EXTERNAL_RESOURCES",    gfx::DeviceFeatures::EXTERNAL_RESOURCES)
        .value("HOST_QUERY_RESET",      gfx::DeviceFeatures::HOST_QUERY_RESET)
        .value("CALIBRATED_TIMESTAMPS", gfx::DeviceFeatures::CALIBRATED_TIMESTAMPS)
        .value("TIMELINE_SEMAPHORES",   gfx::DeviceFeatures::TIMELINE_SEMAPHORES)
        .value("WIDE_LINES",            gfx::DeviceFeatures::WIDE_LINES)
    ;

    nb::enum_<VkPhysicalDeviceType>(m, "PhysicalDeviceType")
        .value("OTHER",          VK_PHYSICAL_DEVICE_TYPE_OTHER)
        .value("INTEGRATED_GPU", VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
        .value("DISCRETE_GPU",   VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        .value("VIRTUAL_GPU",    VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU)
        .value("CPU",            VK_PHYSICAL_DEVICE_TYPE_CPU)
    ;

    nb::class_<DeviceSparseProperties>(m, "DeviceSparseProperties",
        nb::intrusive_ptr<DeviceSparseProperties>([](DeviceSparseProperties *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def_prop_ro("residency_standard_2d_block_shape"             , [](DeviceSparseProperties& sparse_properties) { return (bool)sparse_properties.sparse_properties.residencyStandard2DBlockShape; })
        .def_prop_ro("residency_standard_2d_multisample_block_shape" , [](DeviceSparseProperties& sparse_properties) { return (bool)sparse_properties.sparse_properties.residencyStandard2DMultisampleBlockShape; })
        .def_prop_ro("residency_standard_3d_block_shape"             , [](DeviceSparseProperties& sparse_properties) { return (bool)sparse_properties.sparse_properties.residencyStandard3DBlockShape; })
        .def_prop_ro("residency_aligned_mip_size"                    , [](DeviceSparseProperties& sparse_properties) { return (bool)sparse_properties.sparse_properties.residencyAlignedMipSize; })
        .def_prop_ro("residency_non_resident_strict"                 , [](DeviceSparseProperties& sparse_properties) { return (bool)sparse_properties.sparse_properties.residencyNonResidentStrict; })
    ;

    nb::class_<DeviceLimits>(m, "DeviceLimits",
        nb::intrusive_ptr<DeviceLimits>([](DeviceLimits *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def_prop_ro("max_image_dimension_1d"                               , [](DeviceLimits& limits) { return limits.limits.maxImageDimension1D; })
        .def_prop_ro("max_image_dimension_2d"                               , [](DeviceLimits& limits) { return limits.limits.maxImageDimension2D; })
        .def_prop_ro("max_image_dimension_3d"                               , [](DeviceLimits& limits) { return limits.limits.maxImageDimension3D; })
        .def_prop_ro("max_image_dimension_cube"                             , [](DeviceLimits& limits) { return limits.limits.maxImageDimensionCube; })
        .def_prop_ro("max_image_array_layers"                               , [](DeviceLimits& limits) { return limits.limits.maxImageArrayLayers; })
        .def_prop_ro("max_texel_buffer_elements"                            , [](DeviceLimits& limits) { return limits.limits.maxTexelBufferElements; })
        .def_prop_ro("max_uniform_buffer_range"                             , [](DeviceLimits& limits) { return limits.limits.maxUniformBufferRange; })
        .def_prop_ro("max_storage_buffer_range"                             , [](DeviceLimits& limits) { return limits.limits.maxStorageBufferRange; })
        .def_prop_ro("max_push_constants_size"                              , [](DeviceLimits& limits) { return limits.limits.maxPushConstantsSize; })
        .def_prop_ro("max_memory_allocation_count"                          , [](DeviceLimits& limits) { return limits.limits.maxMemoryAllocationCount; })
        .def_prop_ro("max_sampler_allocation_count"                         , [](DeviceLimits& limits) { return limits.limits.maxSamplerAllocationCount; })
        .def_prop_ro("buffer_image_granularity"                             , [](DeviceLimits& limits) { return limits.limits.bufferImageGranularity; })
        .def_prop_ro("sparse_address_space_size"                            , [](DeviceLimits& limits) { return limits.limits.sparseAddressSpaceSize; })
        .def_prop_ro("max_bound_descriptor_sets"                            , [](DeviceLimits& limits) { return limits.limits.maxBoundDescriptorSets; })
        .def_prop_ro("max_per_stage_descriptor_samplers"                    , [](DeviceLimits& limits) { return limits.limits.maxPerStageDescriptorSamplers; })
        .def_prop_ro("max_per_stage_descriptor_uniform_buffers"             , [](DeviceLimits& limits) { return limits.limits.maxPerStageDescriptorUniformBuffers; })
        .def_prop_ro("max_per_stage_descriptor_storage_buffers"             , [](DeviceLimits& limits) { return limits.limits.maxPerStageDescriptorStorageBuffers; })
        .def_prop_ro("max_per_stage_descriptor_sampled_images"              , [](DeviceLimits& limits) { return limits.limits.maxPerStageDescriptorSampledImages; })
        .def_prop_ro("max_per_stage_descriptor_storage_images"              , [](DeviceLimits& limits) { return limits.limits.maxPerStageDescriptorStorageImages; })
        .def_prop_ro("max_per_stage_descriptor_input_attachments"           , [](DeviceLimits& limits) { return limits.limits.maxPerStageDescriptorInputAttachments; })
        .def_prop_ro("max_per_stage_resources"                              , [](DeviceLimits& limits) { return limits.limits.maxPerStageResources; })
        .def_prop_ro("max_descriptor_set_samplers"                          , [](DeviceLimits& limits) { return limits.limits.maxDescriptorSetSamplers; })
        .def_prop_ro("max_descriptor_set_uniform_buffers"                   , [](DeviceLimits& limits) { return limits.limits.maxDescriptorSetUniformBuffers; })
        .def_prop_ro("max_descriptor_set_uniform_buffers_dynamic"           , [](DeviceLimits& limits) { return limits.limits.maxDescriptorSetUniformBuffersDynamic; })
        .def_prop_ro("max_descriptor_set_storage_buffers"                   , [](DeviceLimits& limits) { return limits.limits.maxDescriptorSetStorageBuffers; })
        .def_prop_ro("max_descriptor_set_storage_buffers_dynamic"           , [](DeviceLimits& limits) { return limits.limits.maxDescriptorSetStorageBuffersDynamic; })
        .def_prop_ro("max_descriptor_set_sampled_images"                    , [](DeviceLimits& limits) { return limits.limits.maxDescriptorSetSampledImages; })
        .def_prop_ro("max_descriptor_set_storage_images"                    , [](DeviceLimits& limits) { return limits.limits.maxDescriptorSetStorageImages; })
        .def_prop_ro("max_descriptor_set_input_attachments"                 , [](DeviceLimits& limits) { return limits.limits.maxDescriptorSetInputAttachments; })
        .def_prop_ro("max_vertex_input_attributes"                          , [](DeviceLimits& limits) { return limits.limits.maxVertexInputAttributes; })
        .def_prop_ro("max_vertex_input_bindings"                            , [](DeviceLimits& limits) { return limits.limits.maxVertexInputBindings; })
        .def_prop_ro("max_vertex_input_attribute_offset"                    , [](DeviceLimits& limits) { return limits.limits.maxVertexInputAttributeOffset; })
        .def_prop_ro("max_vertex_input_binding_stride"                      , [](DeviceLimits& limits) { return limits.limits.maxVertexInputBindingStride; })
        .def_prop_ro("max_vertex_output_components"                         , [](DeviceLimits& limits) { return limits.limits.maxVertexOutputComponents; })
        .def_prop_ro("max_tessellation_generation_level"                    , [](DeviceLimits& limits) { return limits.limits.maxTessellationGenerationLevel; })
        .def_prop_ro("max_tessellation_patch_size"                          , [](DeviceLimits& limits) { return limits.limits.maxTessellationPatchSize; })
        .def_prop_ro("max_tessellation_control_per_vertex_input_components" , [](DeviceLimits& limits) { return limits.limits.maxTessellationControlPerVertexInputComponents; })
        .def_prop_ro("max_tessellation_control_per_vertex_output_components", [](DeviceLimits& limits) { return limits.limits.maxTessellationControlPerVertexOutputComponents; })
        .def_prop_ro("max_tessellation_control_per_patch_output_components" , [](DeviceLimits& limits) { return limits.limits.maxTessellationControlPerPatchOutputComponents; })
        .def_prop_ro("max_tessellation_control_total_output_components"     , [](DeviceLimits& limits) { return limits.limits.maxTessellationControlTotalOutputComponents; })
        .def_prop_ro("max_tessellation_evaluation_input_components"         , [](DeviceLimits& limits) { return limits.limits.maxTessellationEvaluationInputComponents; })
        .def_prop_ro("max_tessellation_evaluation_output_components"        , [](DeviceLimits& limits) { return limits.limits.maxTessellationEvaluationOutputComponents; })
        .def_prop_ro("max_geometry_shader_invocations"                      , [](DeviceLimits& limits) { return limits.limits.maxGeometryShaderInvocations; })
        .def_prop_ro("max_geometry_input_components"                        , [](DeviceLimits& limits) { return limits.limits.maxGeometryInputComponents; })
        .def_prop_ro("max_geometry_output_components"                       , [](DeviceLimits& limits) { return limits.limits.maxGeometryOutputComponents; })
        .def_prop_ro("max_geometry_output_vertices"                         , [](DeviceLimits& limits) { return limits.limits.maxGeometryOutputVertices; })
        .def_prop_ro("max_geometry_total_output_components"                 , [](DeviceLimits& limits) { return limits.limits.maxGeometryTotalOutputComponents; })
        .def_prop_ro("max_fragment_input_components"                        , [](DeviceLimits& limits) { return limits.limits.maxFragmentInputComponents; })
        .def_prop_ro("max_fragment_output_attachments"                      , [](DeviceLimits& limits) { return limits.limits.maxFragmentOutputAttachments; })
        .def_prop_ro("max_fragment_dual_src_attachments"                    , [](DeviceLimits& limits) { return limits.limits.maxFragmentDualSrcAttachments; })
        .def_prop_ro("max_fragment_combined_output_resources"               , [](DeviceLimits& limits) { return limits.limits.maxFragmentCombinedOutputResources; })
        .def_prop_ro("max_compute_shared_memory_size"                       , [](DeviceLimits& limits) { return limits.limits.maxComputeSharedMemorySize; })
        .def_prop_ro("max_compute_work_group_count"                         , [](DeviceLimits& limits) { return nb::make_tuple(limits.limits.maxComputeWorkGroupCount[0], limits.limits.maxComputeWorkGroupCount[1], limits.limits.maxComputeWorkGroupCount[2]); })
        .def_prop_ro("max_compute_work_group_invocations"                   , [](DeviceLimits& limits) { return limits.limits.maxComputeWorkGroupInvocations; })
        .def_prop_ro("max_compute_work_group_size"                          , [](DeviceLimits& limits) { return nb::make_tuple(limits.limits.maxComputeWorkGroupSize[0], limits.limits.maxComputeWorkGroupSize[1], limits.limits.maxComputeWorkGroupSize[2]); })
        .def_prop_ro("sub_pixel_precision_bits"                             , [](DeviceLimits& limits) { return limits.limits.subPixelPrecisionBits; })
        .def_prop_ro("sub_texel_precision_bits"                             , [](DeviceLimits& limits) { return limits.limits.subTexelPrecisionBits; })
        .def_prop_ro("mipmap_precision_bits"                                , [](DeviceLimits& limits) { return limits.limits.mipmapPrecisionBits; })
        .def_prop_ro("max_draw_indexed_index_value"                         , [](DeviceLimits& limits) { return limits.limits.maxDrawIndexedIndexValue; })
        .def_prop_ro("max_draw_indirect_count"                              , [](DeviceLimits& limits) { return limits.limits.maxDrawIndirectCount; })
        .def_prop_ro("max_sampler_lod_bias"                                 , [](DeviceLimits& limits) { return limits.limits.maxSamplerLodBias; })
        .def_prop_ro("max_sampler_anisotropy"                               , [](DeviceLimits& limits) { return limits.limits.maxSamplerAnisotropy; })
        .def_prop_ro("max_viewports"                                        , [](DeviceLimits& limits) { return limits.limits.maxViewports; })
        .def_prop_ro("max_viewport_dimensions"                              , [](DeviceLimits& limits) { return nb::make_tuple(limits.limits.maxViewportDimensions[0], limits.limits.maxViewportDimensions[1]); })
        .def_prop_ro("viewport_bounds_range"                                , [](DeviceLimits& limits) { return nb::make_tuple(limits.limits.viewportBoundsRange[0], limits.limits.viewportBoundsRange[1]); })
        .def_prop_ro("viewport_sub_pixel_bits"                              , [](DeviceLimits& limits) { return limits.limits.viewportSubPixelBits; })
        .def_prop_ro("min_memory_map_alignment"                             , [](DeviceLimits& limits) { return limits.limits.minMemoryMapAlignment; })
        .def_prop_ro("min_texel_buffer_offset_alignment"                    , [](DeviceLimits& limits) { return limits.limits.minTexelBufferOffsetAlignment; })
        .def_prop_ro("min_uniform_buffer_offset_alignment"                  , [](DeviceLimits& limits) { return limits.limits.minUniformBufferOffsetAlignment; })
        .def_prop_ro("min_storage_buffer_offset_alignment"                  , [](DeviceLimits& limits) { return limits.limits.minStorageBufferOffsetAlignment; })
        .def_prop_ro("min_texel_offset"                                     , [](DeviceLimits& limits) { return limits.limits.minTexelOffset; })
        .def_prop_ro("max_texel_offset"                                     , [](DeviceLimits& limits) { return limits.limits.maxTexelOffset; })
        .def_prop_ro("min_texel_gather_offset"                              , [](DeviceLimits& limits) { return limits.limits.minTexelGatherOffset; })
        .def_prop_ro("max_texel_gather_offset"                              , [](DeviceLimits& limits) { return limits.limits.maxTexelGatherOffset; })
        .def_prop_ro("min_interpolation_offset"                             , [](DeviceLimits& limits) { return limits.limits.minInterpolationOffset; })
        .def_prop_ro("max_interpolation_offset"                             , [](DeviceLimits& limits) { return limits.limits.maxInterpolationOffset; })
        .def_prop_ro("sub_pixel_interpolation_offset_bits"                  , [](DeviceLimits& limits) { return limits.limits.subPixelInterpolationOffsetBits; })
        .def_prop_ro("max_framebuffer_width"                                , [](DeviceLimits& limits) { return limits.limits.maxFramebufferWidth; })
        .def_prop_ro("max_framebuffer_height"                               , [](DeviceLimits& limits) { return limits.limits.maxFramebufferHeight; })
        .def_prop_ro("max_framebuffer_layers"                               , [](DeviceLimits& limits) { return limits.limits.maxFramebufferLayers; })
        .def_prop_ro("framebuffer_color_sample_counts"                      , [](DeviceLimits& limits) { return limits.limits.framebufferColorSampleCounts; })
        .def_prop_ro("framebuffer_depth_sample_counts"                      , [](DeviceLimits& limits) { return limits.limits.framebufferDepthSampleCounts; })
        .def_prop_ro("framebuffer_stencil_sample_counts"                    , [](DeviceLimits& limits) { return limits.limits.framebufferStencilSampleCounts; })
        .def_prop_ro("framebuffer_no_attachments_sample_counts"             , [](DeviceLimits& limits) { return limits.limits.framebufferNoAttachmentsSampleCounts; })
        .def_prop_ro("max_color_attachments"                                , [](DeviceLimits& limits) { return limits.limits.maxColorAttachments; })
        .def_prop_ro("sampled_image_color_sample_counts"                    , [](DeviceLimits& limits) { return limits.limits.sampledImageColorSampleCounts; })
        .def_prop_ro("sampled_image_integer_sample_counts"                  , [](DeviceLimits& limits) { return limits.limits.sampledImageIntegerSampleCounts; })
        .def_prop_ro("sampled_image_depth_sample_counts"                    , [](DeviceLimits& limits) { return limits.limits.sampledImageDepthSampleCounts; })
        .def_prop_ro("sampled_image_stencil_sample_counts"                  , [](DeviceLimits& limits) { return limits.limits.sampledImageStencilSampleCounts; })
        .def_prop_ro("storage_image_sample_counts"                          , [](DeviceLimits& limits) { return limits.limits.storageImageSampleCounts; })
        .def_prop_ro("max_sample_mask_words"                                , [](DeviceLimits& limits) { return limits.limits.maxSampleMaskWords; })
        .def_prop_ro("timestamp_compute_and_graphics"                       , [](DeviceLimits& limits) { return limits.limits.timestampComputeAndGraphics; })
        .def_prop_ro("timestamp_period"                                     , [](DeviceLimits& limits) { return limits.limits.timestampPeriod; })
        .def_prop_ro("max_clip_distances"                                   , [](DeviceLimits& limits) { return limits.limits.maxClipDistances; })
        .def_prop_ro("max_cull_distances"                                   , [](DeviceLimits& limits) { return limits.limits.maxCullDistances; })
        .def_prop_ro("max_combined_clip_and_cull_distances"                 , [](DeviceLimits& limits) { return limits.limits.maxCombinedClipAndCullDistances; })
        .def_prop_ro("discrete_queue_priorities"                            , [](DeviceLimits& limits) { return limits.limits.discreteQueuePriorities; })
        .def_prop_ro("point_size_range"                                     , [](DeviceLimits& limits) { return nb::make_tuple(limits.limits.pointSizeRange[0], limits.limits.pointSizeRange[1]); })
        .def_prop_ro("line_width_range"                                     , [](DeviceLimits& limits) { return nb::make_tuple(limits.limits.lineWidthRange[0], limits.limits.lineWidthRange[1]); })
        .def_prop_ro("point_size_granularity"                               , [](DeviceLimits& limits) { return limits.limits.pointSizeGranularity; })
        .def_prop_ro("line_width_granularity"                               , [](DeviceLimits& limits) { return limits.limits.lineWidthGranularity; })
        .def_prop_ro("strict_lines"                                         , [](DeviceLimits& limits) { return limits.limits.strictLines; })
        .def_prop_ro("standard_sample_locations"                            , [](DeviceLimits& limits) { return limits.limits.standardSampleLocations; })
        .def_prop_ro("optimal_buffer_copy_offset_alignment"                 , [](DeviceLimits& limits) { return limits.limits.optimalBufferCopyOffsetAlignment; })
        .def_prop_ro("optimal_buffer_copy_row_pitch_alignment"              , [](DeviceLimits& limits) { return limits.limits.optimalBufferCopyRowPitchAlignment; })
        .def_prop_ro("non_coherent_atom_size"                               , [](DeviceLimits& limits) { return limits.limits.nonCoherentAtomSize; })
    ;

    nb::class_<DeviceProperties>(m, "DeviceProperties",
        nb::intrusive_ptr<DeviceProperties>([](DeviceProperties *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def_prop_ro("limits", [](DeviceProperties& properties) {
            return properties.limits;
        })
        .def_prop_ro("sparse_properties", [](DeviceProperties& properties) {
            return properties.sparse_properties;
        })
        .def_ro("api_version", &DeviceProperties::api_version)
        .def_ro("driver_version", &DeviceProperties::driver_version)
        .def_ro("vendor_id", &DeviceProperties::vendor_id)
        .def_ro("device_id", &DeviceProperties::device_id)
        .def_ro("device_type", &DeviceProperties::device_type)
        .def_ro("device_name", &DeviceProperties::device_name)
        .def_prop_ro("pipeline_cache_uuid", [](DeviceProperties& properties) {
            return nb::bytes(properties.pipeline_cache_uuid, VK_UUID_SIZE);
        })
    ;

    nb::class_<Context>(m, "Context",
        nb::intrusive_ptr<Context>([](Context *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<std::tuple<u32, u32>, gfx::DeviceFeatures::Flags, gfx::DeviceFeatures::Flags, bool, u32, bool, u32, bool, bool, bool, bool, bool>(),
            nb::arg("version") = std::make_tuple(1, 1),
            nb::arg("required_features") = gfx::DeviceFeatures::Flags(gfx::DeviceFeatures::DYNAMIC_RENDERING | gfx::DeviceFeatures::SYNCHRONIZATION_2),
            nb::arg("optional_features") = gfx::DeviceFeatures::Flags(gfx::DeviceFeatures::NONE),
            nb::arg("presentation") = true,
            nb::arg("preferred_frames_in_flight") = 2,
            nb::arg("vsync") = true,
            nb::arg("force_physical_device_index") = ~0U,
            nb::arg("prefer_discrete_gpu") = true,
            nb::arg("enable_debug_utils") = false,
            nb::arg("enable_validation_layer") = false,
            nb::arg("enable_gpu_based_validation") = false,
            nb::arg("enable_synchronization_validation") = false
        )
        .def("sync_commands", [] (nb::ref<Context> ctx) {
            return new CommandsManager(new CommandBuffer(ctx, ctx->vk.sync_command_pool, ctx->vk.sync_command_buffer), ctx->vk.queue, {}, {}, {}, {}, ctx->vk.sync_fence, true);
        })
        .def_prop_ro("sync_command_buffer", [] (nb::ref<Context> ctx) {
            return new CommandBuffer(ctx, ctx->vk.sync_command_pool, ctx->vk.sync_command_buffer);
        })
        .def("submit_sync", [](const Context& ctx) {
            gfx::SubmitSync(ctx.vk);
        })
        .def("wait_idle", [](Context& ctx) {
            nb::gil_scoped_release gil;
            gfx::WaitIdle(ctx.vk);
        })
        .def_prop_ro("instance_version", [](Context& ctx) {
            return nb::make_tuple(VK_API_VERSION_MAJOR(ctx.vk.instance_version), VK_API_VERSION_MINOR(ctx.vk.instance_version));
        })
        .def_prop_ro("version", [](Context& ctx) {
            return nb::make_tuple(VK_API_VERSION_MAJOR(ctx.vk.device_version), VK_API_VERSION_MINOR(ctx.vk.device_version));
        })
        .def_prop_ro("device_features", [](Context& ctx) {
            return ctx.vk.device_features.flags;
        })
        .def_ro("device_properties", &Context::device_properties)
        .def_ro("memory_properties", &Context::memory_properties)
        .def_prop_ro("heap_statistics", [](Context& ctx) {
            VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
            vmaGetHeapBudgets(ctx.vk.vma, budgets);

            usize heap_count = ctx.memory_properties->memory_heaps.size();
            std::vector<nb::ref<HeapStatistics>> statistics;
            statistics.reserve(heap_count);
            for (size_t i = 0; i < heap_count; i++) {
                statistics.push_back(new HeapStatistics(budgets[i]));
            }
            return statistics;
        }, nb::rv_policy::move)
        .def_prop_ro("has_compute_queue", [](Context& ctx) {
            return ctx.vk.compute_queue != VK_NULL_HANDLE;
        })
        .def_prop_ro("has_transfer_queue", [](Context& ctx) {
            return ctx.vk.copy_queue != VK_NULL_HANDLE;
        })
        .def_prop_ro("graphics_queue_family_index", [](Context& ctx) {
            return ctx.vk.queue_family_index;
        })
        .def_prop_ro("compute_queue_family_index", [](Context& ctx) {
            if (ctx.vk.compute_queue == VK_NULL_HANDLE) {
                throw std::runtime_error("Compute queue not supported by device");
            }
            return ctx.vk.compute_queue_family_index;
        })
        .def_prop_ro("transfer_queue_family_index", [](Context& ctx) {
            if (ctx.vk.copy_queue == VK_NULL_HANDLE) {
                throw std::runtime_error("Transfer queue not supported by device");
            }
            return ctx.vk.copy_queue_family_index;
        })
        .def_prop_ro("timestamp_period_ns", [](Context& ctx) {
            return ctx.vk.timestamp_period_ns;
        })
        .def("reset_query_pool", [](Context& ctx, const QueryPool& pool) {
            if (!(ctx.vk.device_features & gfx::DeviceFeatures::HOST_QUERY_RESET)) {
                throw std::runtime_error("Device feature HOST_QUERY_RESET must be set to use Context.reset_query_pool");
            }
            vkResetQueryPoolEXT(ctx.vk.device, pool.pool, 0, pool.count);
        })
        .def("get_calibrated_timestamps", [](Context& ctx) -> std::tuple<u64, u64> {
            if (!(ctx.vk.device_features & gfx::DeviceFeatures::CALIBRATED_TIMESTAMPS)) {
                throw std::runtime_error("Device feature HOST_QUERY_RESET must be set to use Context.get_calibrated_timestamps");
            }

            VkCalibratedTimestampInfoKHR timestamp_infos[2] = {};
            timestamp_infos[0].sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT;
#ifdef _WIN32
            timestamp_infos[0].timeDomain = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT;
#else
            timestamp_infos[0].timeDomain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR;
#endif
            timestamp_infos[1].sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT;
            timestamp_infos[1].timeDomain = VK_TIME_DOMAIN_DEVICE_KHR;

            u64 timestamps[2];
            u64 deviations[2];
            vkGetCalibratedTimestampsEXT(ctx.vk.device, 2, timestamp_infos, timestamps, deviations);

#ifdef _WIN32
            // Convert performance counter ticks to ns. This matches what time.perf_counter_ns does.
            static LARGE_INTEGER frequency = {};
            if (!frequency.QuadPart) {
                QueryPerformanceFrequency(&frequency);
            }
            timestamps[0] = mul_div(timestamps[0], 1000000000, frequency.QuadPart);
#else
            // On linux timestamp is already in nanoseconds. Same as time.perf_counter_ns.
#endif

            return std::make_tuple(timestamps[0], timestamps[1]);
        })
        .def_prop_ro("queue", [](nb::ref<Context> ctx) -> nb::ref<Queue> { return new Queue(ctx, ctx->vk.queue); })
        .def_prop_ro("compute_queue", [](nb::ref<Context> ctx) -> nb::ref<Queue> {
            if (ctx->vk.compute_queue == VK_NULL_HANDLE) {
                throw std::runtime_error("Compute queue not supported by device");
            }
            return new Queue(ctx, ctx->vk.compute_queue);
        })
        .def_prop_ro("transfer_queue", [](nb::ref<Context> ctx) -> nb::ref<Queue> {
            if (ctx->vk.copy_queue == VK_NULL_HANDLE) {
                throw std::runtime_error("Transfer queue not supported by device");
            }
            return new Queue(ctx, ctx->vk.copy_queue);
        })
    ;

    nb::class_<CommandsManager>(m, "CommandsManager",
        nb::intrusive_ptr<CommandsManager>([](CommandsManager *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def("__enter__", &CommandsManager::enter)
        .def("__exit__", &CommandsManager::exit, nb::arg("exc_type").none(), nb::arg("exc_val").none(), nb::arg("exc_tb").none())
    ;

    nb::class_<Frame>(m, "Frame",
        nb::intrusive_ptr<Frame>([](Frame *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def_ro("command_buffer", &Frame::command_buffer)
        .def_ro("image", &Frame::image)
        .def("compute_commands", [](Frame& frame, std::vector<std::tuple<nb::ref<Semaphore>, VkPipelineStageFlagBits>> wait_semaphores, std::vector<u64> wait_timeline_values, std::vector<nb::ref<Semaphore>> signal_semaphores, std::vector<u64> signal_timeline_values) {
            if (!frame.compute_command_buffer.has_value()) {
                nb::raise("Device does not support compute queue. Check Context.has_compute_queue to know if it's supported.");
            }
            return new CommandsManager(frame.compute_command_buffer.value(), frame.window->ctx->vk.compute_queue, std::move(wait_semaphores), std::move(wait_timeline_values), std::move(signal_semaphores), std::move(signal_timeline_values), VK_NULL_HANDLE, false);
        }, nb::arg("wait_semaphores") = nb::list(), nb::arg("wait_timeline_values") = nb::list(), nb::arg("signal_semaphores") = nb::list(), nb::arg("signal_timeline_values") = nb::list())
        .def_ro("compute_command_buffer", &Frame::compute_command_buffer)
        .def("transfer_commands", [](Frame& frame, std::vector<std::tuple<nb::ref<Semaphore>, VkPipelineStageFlagBits>> wait_semaphores, std::vector<u64> wait_timeline_values, std::vector<nb::ref<Semaphore>> signal_semaphores, std::vector<u64> signal_timeline_values) {
            if (!frame.transfer_command_buffer.has_value()) {
                nb::raise("Device does not support transfer queue. Check Context.has_transfer_queue to know if it's supported.");
            }
            return new CommandsManager(frame.transfer_command_buffer.value(), frame.window->ctx->vk.copy_queue, std::move(wait_semaphores), std::move(wait_timeline_values), std::move(signal_semaphores), std::move(signal_timeline_values), VK_NULL_HANDLE, false);
        }, nb::arg("wait_semaphores") = nb::list(), nb::arg("wait_timeline_values") = nb::list(), nb::arg("signal_semaphores") = nb::list(), nb::arg("signal_timeline_values") = nb::list())
        .def_ro("transfer_command_buffer", &Frame::transfer_command_buffer)
    ;

    nb::class_<Window>(m, "Window",
        nb::type_slots(window_tp_slots),
        nb::intrusive_ptr<Window>([](Window *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<nb::ref<Context>, const::std::string&, u32, u32, std::optional<u32>, std::optional<u32>>(), nb::arg("ctx"), nb::arg("title"), nb::arg("width"), nb::arg("height"), nb::arg("x") = nb::none(), nb::arg("y") = nb::none())
        .def("should_close", &Window::should_close)
        .def("get_modifiers_state", &Window::get_modifiers_state)
        .def("set_callbacks", &Window::set_callbacks,
            nb::arg("draw"),
            nb::arg("mouse_move_event") = nb::none(),
            nb::arg("mouse_button_event") = nb::none(),
            nb::arg("mouse_scroll_event") = nb::none(),
            nb::arg("key_event") = nb::none(),
#if PY_MINOR_VERSION >= 9
            nb::sig("def set_callbacks(self, draw: Callable[[], None], mouse_move_event: Callable[[tuple[int, int]], None] | None = None, mouse_button_event: Callable[[tuple[int, int], MouseButton, Action, Modifiers], None] | None = None, mouse_scroll_event: Callable[[tuple[int, int], tuple[int, int]], None] | None = None, key_event: Callable[[Key, Action, Modifiers], None] | None = None) -> None")
#else
            nb::sig("def set_callbacks(self, draw: Callable[[], None], mouse_move_event: Callable[[Tuple[int, int]], None] | None = None, mouse_button_event: Callable[[Tuple[int, int], MouseButton, Action, Modifiers], None] | None = None, mouse_scroll_event: Callable[[Tuple[int, int], Tuple[int, int]], None] | None = None, key_event: Callable[[Key, Action, Modifiers], None] | None = None) -> None")
#endif
        )
        .def("reset_callbacks", &Window::reset_callbacks)
        .def("update_swapchain", &Window::update_swapchain)
        .def("begin_frame", &Window::begin_frame)
        .def("end_frame", &Window::end_frame,
            nb::arg("frame"),
            nb::arg("additional_wait_semaphores") = nb::list(),
            nb::arg("additional_wait_timeline_values") = nb::list(),
            nb::arg("additional_signal_semaphores") = nb::list(),
            nb::arg("additional_signal_timeline_values") = nb::list()
        )
        .def("frame", &Window::frame,
            nb::arg("additional_wait_semaphores") = nb::list(),
            nb::arg("additional_wait_timeline_values") = nb::list(),
            nb::arg("additional_signal_semaphores") = nb::list(),
            nb::arg("additional_signal_timeline_values") = nb::list()
        )
        .def("post_empty_event", &Window::post_empty_event)
        .def_prop_ro("swapchain_format", [](Window& w) -> VkFormat { return w.window.swapchain_format; })
        .def_prop_ro("num_swapchain_images", [](Window& w) -> usize { return w.window.image_views.length; })
        .def_prop_ro("fb_width", [](Window& w) -> u32 { return w.window.fb_width; })
        .def_prop_ro("fb_height", [](Window& w) -> u32 { return w.window.fb_height; })
        .def_prop_ro("num_frames", [](Window& w) -> usize { return w.window.frames.length; })
    ;

    nb::class_<Window::FrameManager>(m, "WindowFrame",
        nb::intrusive_ptr<Window::FrameManager>([](Window::FrameManager *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def("__enter__", &Window::FrameManager::enter)
        .def("__exit__", &Window::FrameManager::exit, nb::arg("exc_type").none(), nb::arg("exc_val").none(), nb::arg("exc_tb").none())
    ;

    nb::class_<Gui::GuiFrame>(m, "GuiFrame")
        .def("__enter__", &Gui::GuiFrame::enter)
        .def("__exit__", &Gui::GuiFrame::exit, nb::arg("exc_type").none(), nb::arg("exc_val").none(), nb::arg("exc_tb").none())
    ;

    nb::class_<Gui>(m, "Gui",
        nb::type_slots(gui_tp_slots),
        nb::intrusive_ptr<Gui>([](Gui *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<nb::ref<Window>>(), nb::arg("window"))
        .def("begin_frame", &Gui::begin_frame)
        .def("end_frame", &Gui::end_frame)
        .def("render", &Gui::render, nb::arg("frame"))
        .def("frame", &Gui::frame)
        .def("set_ini_filename", &Gui::set_ini_filename)
    ;

    nb::enum_<gfx::Action>(m, "Action")
        .value("NONE", gfx::Action::None)
        .value("RELEASE", gfx::Action::Release)
        .value("PRESS", gfx::Action::Press)
        .value("REPEAT", gfx::Action::Repeat)
    ;

    nb::enum_<gfx::Key>(m, "Key")
        .value("SPACE",                 gfx::Key::Space)
        .value("APOSTROPHE",            gfx::Key::Apostrophe)
        .value("COMMA",                 gfx::Key::Comma)
        .value("MINUS",                 gfx::Key::Minus)
        .value("PERIOD",                gfx::Key::Period)
        .value("SLASH",                 gfx::Key::Slash)
        .value("N0",                    gfx::Key::N0)
        .value("N1",                    gfx::Key::N1)
        .value("N2",                    gfx::Key::N2)
        .value("N3",                    gfx::Key::N3)
        .value("N4",                    gfx::Key::N4)
        .value("N5",                    gfx::Key::N5)
        .value("N6",                    gfx::Key::N6)
        .value("N7",                    gfx::Key::N7)
        .value("N8",                    gfx::Key::N8)
        .value("N9",                    gfx::Key::N9)
        .value("SEMICOLON",             gfx::Key::Semicolon)
        .value("EQUAL",                 gfx::Key::Equal)
        .value("A",                     gfx::Key::A)
        .value("B",                     gfx::Key::B)
        .value("C",                     gfx::Key::C)
        .value("D",                     gfx::Key::D)
        .value("E",                     gfx::Key::E)
        .value("F",                     gfx::Key::F)
        .value("G",                     gfx::Key::G)
        .value("H",                     gfx::Key::H)
        .value("I",                     gfx::Key::I)
        .value("J",                     gfx::Key::J)
        .value("K",                     gfx::Key::K)
        .value("L",                     gfx::Key::L)
        .value("M",                     gfx::Key::M)
        .value("N",                     gfx::Key::N)
        .value("O",                     gfx::Key::O)
        .value("P",                     gfx::Key::P)
        .value("Q",                     gfx::Key::Q)
        .value("R",                     gfx::Key::R)
        .value("S",                     gfx::Key::S)
        .value("T",                     gfx::Key::T)
        .value("U",                     gfx::Key::U)
        .value("V",                     gfx::Key::V)
        .value("W",                     gfx::Key::W)
        .value("X",                     gfx::Key::X)
        .value("Y",                     gfx::Key::Y)
        .value("Z",                     gfx::Key::Z)
        .value("LEFT_BRACKET",          gfx::Key::LeftBracket)
        .value("BACKSLASH",             gfx::Key::Backslash)
        .value("RIGHT_BRACKET",         gfx::Key::RightBracket)
        .value("GRAVE_ACCENT",          gfx::Key::GraveAccent)
        .value("WORLD_1",               gfx::Key::World1)
        .value("WORLD_2",               gfx::Key::World2)
        .value("ESCAPE",                gfx::Key::Escape)
        .value("ENTER",                 gfx::Key::Enter)
        .value("TAB",                   gfx::Key::Tab)
        .value("BACKSPACE",             gfx::Key::Backspace)
        .value("INSERT",                gfx::Key::Insert)
        .value("DELETE",                gfx::Key::Delete)
        .value("RIGHT",                 gfx::Key::Right)
        .value("LEFT",                  gfx::Key::Left)
        .value("DOWN",                  gfx::Key::Down)
        .value("UP",                    gfx::Key::Up)
        .value("PAGE_UP",               gfx::Key::PageUp)
        .value("PAGE_DOWN",             gfx::Key::PageDown)
        .value("HOME",                  gfx::Key::Home)
        .value("END",                   gfx::Key::End)
        .value("CAPS_LOCK",             gfx::Key::CapsLock)
        .value("SCROLL_LOCK",           gfx::Key::ScrollLock)
        .value("NUM_LOCK",              gfx::Key::NumLock)
        .value("PRINT_SCREEN",          gfx::Key::PrintScreen)
        .value("PAUSE",                 gfx::Key::Pause)
        .value("F1",                    gfx::Key::F1)
        .value("F2",                    gfx::Key::F2)
        .value("F3",                    gfx::Key::F3)
        .value("F4",                    gfx::Key::F4)
        .value("F5",                    gfx::Key::F5)
        .value("F6",                    gfx::Key::F6)
        .value("F7",                    gfx::Key::F7)
        .value("F8",                    gfx::Key::F8)
        .value("F9",                    gfx::Key::F9)
        .value("F10",                   gfx::Key::F10)
        .value("F11",                   gfx::Key::F11)
        .value("F12",                   gfx::Key::F12)
        .value("F13",                   gfx::Key::F13)
        .value("F14",                   gfx::Key::F14)
        .value("F15",                   gfx::Key::F15)
        .value("F16",                   gfx::Key::F16)
        .value("F17",                   gfx::Key::F17)
        .value("F18",                   gfx::Key::F18)
        .value("F19",                   gfx::Key::F19)
        .value("F20",                   gfx::Key::F20)
        .value("F21",                   gfx::Key::F21)
        .value("F22",                   gfx::Key::F22)
        .value("F23",                   gfx::Key::F23)
        .value("F24",                   gfx::Key::F24)
        .value("F25",                   gfx::Key::F25)
        .value("KP0",                   gfx::Key::KP0)
        .value("KP1",                   gfx::Key::KP1)
        .value("KP2",                   gfx::Key::KP2)
        .value("KP3",                   gfx::Key::KP3)
        .value("KP4",                   gfx::Key::KP4)
        .value("KP5",                   gfx::Key::KP5)
        .value("KP6",                   gfx::Key::KP6)
        .value("KP7",                   gfx::Key::KP7)
        .value("KP8",                   gfx::Key::KP8)
        .value("KP9",                   gfx::Key::KP9)
        .value("KP_DECIMAL",            gfx::Key::KPDecimal)
        .value("KP_DIVIDE",             gfx::Key::KPDivide)
        .value("KP_MULTIPLY",           gfx::Key::KPMultiply)
        .value("KP_SUBTRACT",           gfx::Key::KPSubtract)
        .value("KP_ADD",                gfx::Key::KPAdd)
        .value("KP_ENTER",              gfx::Key::KPEnter)
        .value("KP_EQUAL",              gfx::Key::KPEqual)
        .value("LEFT_SHIFT",            gfx::Key::LeftShift)
        .value("LEFT_CONTROL",          gfx::Key::LeftControl)
        .value("LEFT_ALT",              gfx::Key::LeftAlt)
        .value("LEFT_SUPER",            gfx::Key::LeftSuper)
        .value("RIGHT_SHIFT",           gfx::Key::RightShift)
        .value("RIGHT_CONTROL",         gfx::Key::RightControl)
        .value("RIGHT_ALT",             gfx::Key::RightAlt)
        .value("RIGHT_SUPER",           gfx::Key::RightSuper)
        .value("MENU",                  gfx::Key::Menu)
    ;

    nb::enum_<gfx::MouseButton>(m, "MouseButton")
        .value("NONE",   gfx::MouseButton::None)
        .value("LEFT",   gfx::MouseButton::Left)
        .value("RIGHT",  gfx::MouseButton::Right)
        .value("MIDDLE", gfx::MouseButton::Middle)
    ;

    nb::enum_<gfx::Modifiers>(m, "Modifiers", nb::is_flag(), nb::is_arithmetic())
        .value("NONE", gfx::Modifiers::None)
        .value("SHIFT", gfx::Modifiers::Shift)
        .value("CTRL", gfx::Modifiers::Ctrl)
        .value("ALT", gfx::Modifiers::Alt)
        .value("SUPER", gfx::Modifiers::Super)
    ;

    nb::enum_<gfx::AllocPresets::Type>(m, "AllocType")
        .value("HOST", gfx::AllocPresets::Type::Host)
        .value("HOST_WRITE_COMBINING", gfx::AllocPresets::Type::HostWriteCombining)
        .value("DEVICE_MAPPED_WITH_FALLBACK", gfx::AllocPresets::Type::DeviceMappedWithFallback)
        .value("DEVICE_MAPPED", gfx::AllocPresets::Type::DeviceMapped)
        .value("DEVICE", gfx::AllocPresets::Type::Device)
        .value("DEVICE_DEDICATED", gfx::AllocPresets::Type::DeviceDedicated)
    ;

    nb::enum_<VkBufferUsageFlagBits>(m, "BufferUsageFlags", nb::is_arithmetic() , nb::is_flag())
        .value("TRANSFER_SRC",                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
        .value("TRANSFER_DST",                   VK_BUFFER_USAGE_TRANSFER_DST_BIT)
        .value("UNIFORM",                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)
        .value("STORAGE",                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        .value("INDEX",                          VK_BUFFER_USAGE_INDEX_BUFFER_BIT)
        .value("VERTEX",                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)
        .value("INDIRECT",                       VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT)
        .value("ACCELERATION_STRUCTURE_INPUT",   VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR)
        .value("ACCELERATION_STRUCTURE_STORAGE", VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR)
        .value("SHADER_DEVICE_ADDRESS",          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
    ;

    nb::enum_<VkImageUsageFlagBits>(m, "ImageUsageFlags", nb::is_arithmetic() , nb::is_flag())
        .value("TRANSFER_SRC",                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
        .value("TRANSFER_DST",                         VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .value("SAMPLED",                              VK_IMAGE_USAGE_SAMPLED_BIT)
        .value("STORAGE",                              VK_IMAGE_USAGE_STORAGE_BIT)
        .value("COLOR_ATTACHMENT",                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
        .value("DEPTH_STENCIL_ATTACHMENT",             VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
        .value("TRANSIENT_ATTACHMENT",                 VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT)
        .value("INPUT_ATTACHMENT",                     VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT)
        .value("VIDEO_DECODE_DST",                     VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR)
        .value("VIDEO_DECODE_SRC",                     VK_IMAGE_USAGE_VIDEO_DECODE_SRC_BIT_KHR)
        .value("VIDEO_DECODE_DPB",                     VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR)
        .value("FRAGMENT_DENSITY_MAP",                 VK_IMAGE_USAGE_FRAGMENT_DENSITY_MAP_BIT_EXT)
        .value("FRAGMENT_SHADING_RATE_ATTACHMENT",     VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR)
        .value("HOST_TRANSFER",                        VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT)
        .value("VIDEO_ENCODE_DST",                     VK_IMAGE_USAGE_VIDEO_ENCODE_DST_BIT_KHR)
        .value("VIDEO_ENCODE_SRC",                     VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR)
        .value("VIDEO_ENCODE_DPB",                     VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR)
        .value("ATTACHMENT_FEEDBACK_LOOP",             VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT)
        .value("INVOCATION_MASK",                      VK_IMAGE_USAGE_INVOCATION_MASK_BIT_HUAWEI)
        .value("SAMPLE_WEIGHT",                        VK_IMAGE_USAGE_SAMPLE_WEIGHT_BIT_QCOM)
        .value("SAMPLE_BLOCK_MATCH",                   VK_IMAGE_USAGE_SAMPLE_BLOCK_MATCH_BIT_QCOM)
        .value("SHADING_RATE_IMAGE",                   VK_IMAGE_USAGE_SHADING_RATE_IMAGE_BIT_NV)
    ;

    nb::enum_<VkCompareOp>(m, "CompareOp")
        .value("NEVER",            VK_COMPARE_OP_NEVER)
        .value("LESS",             VK_COMPARE_OP_LESS)
        .value("EQUAL",            VK_COMPARE_OP_EQUAL)
        .value("LESS_OR_EQUAL",    VK_COMPARE_OP_LESS_OR_EQUAL)
        .value("GREATER",          VK_COMPARE_OP_GREATER)
        .value("NOT_EQUAL",        VK_COMPARE_OP_NOT_EQUAL)
        .value("GREATER_OR_EQUAL", VK_COMPARE_OP_GREATER_OR_EQUAL)
        .value("ALWAYS",           VK_COMPARE_OP_ALWAYS)
    ;

    nb::enum_<VkFilter>(m, "Filter")
        .value("NEAREST", VK_FILTER_NEAREST)
        .value("LINEAR",  VK_FILTER_LINEAR)
        .value("CUBIC",   VK_FILTER_CUBIC_EXT)
    ;

    nb::enum_<VkSamplerMipmapMode>(m, "SamplerMipmapMode")
        .value("NEAREST", VK_SAMPLER_MIPMAP_MODE_NEAREST)
        .value("LINEAR", VK_SAMPLER_MIPMAP_MODE_LINEAR)
    ;

    nb::enum_<VkSamplerAddressMode>(m, "SamplerAddressMode")
        .value("REPEAT", VK_SAMPLER_ADDRESS_MODE_REPEAT)
        .value("MIRRORED_REPEAT", VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT)
        .value("CLAMP_TO_EDGE", VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
        .value("CLAMP_TO_BORDER", VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER)
        .value("MIRROR_CLAMP_TO_EDGE", VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE)
    ;

    nb::class_<GfxObject>(m, "GfxObject",
        nb::intrusive_ptr<GfxObject>([](GfxObject *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def_ro("ctx", &GfxObject::ctx)
    ;

    nb::class_<Queue, GfxObject>(m, "Queue")
        .def("__repr__", [](Queue& queue) {
            return nb::str("Queue()");
        })
        .def("submit", &Queue::submit,
            nb::arg("command_buffer"),
            nb::arg("wait_semaphores") = nb::list(),
            nb::arg("wait_timeline_values") = nb::list(),
            nb::arg("signal_semaphores") = nb::list(),
            nb::arg("signal_timeline_values") = nb::list(),
            nb::arg("fence") = nb::none()
        )
        .def("begin_label", &Queue::begin_label, nb::arg("name"), nb::arg("color") = nb::none())
        .def("end_label", &Queue::end_label)
        .def("insert_label", &Queue::insert_label, nb::arg("name"), nb::arg("color") = nb::none())
    ;

    nb::enum_<VkQueryType>(m, "QueryType")
        .value("OCCLUSION"                                                  , VK_QUERY_TYPE_OCCLUSION)
        .value("PIPELINE_STATISTICS"                                        , VK_QUERY_TYPE_PIPELINE_STATISTICS)
        .value("TIMESTAMP"                                                  , VK_QUERY_TYPE_TIMESTAMP)
        .value("RESULT_STATUS_ONLY"                                         , VK_QUERY_TYPE_RESULT_STATUS_ONLY_KHR)
        .value("TRANSFORM_FEEDBACK_STREAM"                                  , VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT)
        .value("PERFORMANCE_QUERY"                                          , VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR)
        .value("ACCELERATION_STRUCTURE_COMPACTED_SIZE"                      , VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR)
        .value("ACCELERATION_STRUCTURE_SERIALIZATION_SIZE"                  , VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR)
        .value("VIDEO_ENCODE_FEEDBACK"                                      , VK_QUERY_TYPE_VIDEO_ENCODE_FEEDBACK_KHR)
        .value("MESH_PRIMITIVES_GENERATED"                                  , VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT)
        .value("PRIMITIVES_GENERATED"                                       , VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT)
        .value("ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS" , VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR)
        .value("ACCELERATION_STRUCTURE_SIZE"                                , VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR)
        .value("MICROMAP_SERIALIZATION_SIZE"                                , VK_QUERY_TYPE_MICROMAP_SERIALIZATION_SIZE_EXT)
        .value("MICROMAP_COMPACTED_SIZE"                                    , VK_QUERY_TYPE_MICROMAP_COMPACTED_SIZE_EXT)
    ;

    nb::class_<QueryPool, GfxObject>(m, "QueryPool")
        .def(nb::init<nb::ref<Context>, VkQueryType, u32, std::optional<nb::str>>(), nb::arg("ctx"), nb::arg("type"), nb::arg("count"), nb::arg("name") = nb::none())
        .def("__repr__", [](QueryPool& pool) {
            return nb::str("QueryPool(name={}, type={}, count={})").format(pool.name, pool.type, pool.count);
        })
        .def_ro("type", &QueryPool::type)
        .def_ro("count", &QueryPool::count)
        .def("wait_results", &QueryPool::wait_results, nb::arg("first"), nb::arg("count"))
    ;

    nb::class_<AllocInfo>(m, "AllocInfo",
        nb::intrusive_ptr<AllocInfo>([](AllocInfo *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def("__repr__", [](AllocInfo& info) {
            return nb::str("AllocationInfo(memory_type={}, offset={}, size={}, is_dedicated={})").format(info.memory_type, info.offset, info.size, info.is_dedicated);
        })
        .def_ro("memory_type", &AllocInfo::memory_type)
        .def_ro("offset", &AllocInfo::offset)
        .def_ro("size", &AllocInfo::size)
        .def_ro("is_dedicated", &AllocInfo::is_dedicated)
    ;

    nb::class_<Buffer, GfxObject> buffer_type(m, "Buffer", nb::type_slots(buffer_slots));
    buffer_type
        .def(nb::init<nb::ref<Context>, size_t, VkBufferUsageFlagBits, gfx::AllocPresets::Type, std::optional<nb::str>>(), nb::arg("ctx"), nb::arg("size"), nb::arg("usage_flags"), nb::arg("alloc_type"), nb::arg("name") = nb::none())
        .def("__repr__", [](Buffer& buf) {
            return nb::str("Buffer(name={}, size={})").format(buf.name, buf.size);
        })
        .def("destroy", &Buffer::destroy)
        .def_static("from_data", &Buffer::from_data, nb::arg("ctx"), nb::arg("data"), nb::arg("usage_flags"), nb::arg("alloc_type"), nb::arg("name") = nb::none())
        .def_prop_ro("data", [] (Buffer& buffer) {
            return nb::steal(PyMemoryView_FromObject(buffer.self_py()));
        }, nb::sig("def data(self) -> memoryview"))
        .def_prop_ro("is_mapped", [](Buffer& buf) {
            return buf.buffer.map.data != 0;
        })
        .def_prop_ro("address", [](Buffer& buffer) {
            if (!buffer.device_address.has_value()) {
                throw std::runtime_error("Buffer address can only be accessed if BufferUsageFlags.SHADER_DEVICE_ADDRESS was set when creating the buffer");
            }
            return buffer.device_address.value();
        })
        .def_ro("size", &Buffer::size)
        .def_ro("alloc", &Buffer::alloc)
    ;

#if PY_VERSION_HEX < 0x03090000
    {
        PyTypeObject* tp = (PyTypeObject*)buffer_type.ptr();
        tp->tp_as_buffer->bf_getbuffer = Buffer::bf_getbuffer;
        tp->tp_as_buffer->bf_releasebuffer = Buffer::bf_releasebuffer;
    }
#endif

    nb::class_<ExternalBuffer, Buffer>(m, "ExternalBuffer")
        .def(nb::init<nb::ref<Context>, size_t, VkBufferUsageFlagBits, gfx::AllocPresets::Type, std::optional<nb::str>>(), nb::arg("ctx"), nb::arg("size"), nb::arg("usage_flags"), nb::arg("alloc_type"), nb::arg("name") = nb::none())
        .def("__repr__", [](ExternalBuffer& buf) {
            return nb::str("ExternalBuffer(name={}, size={})").format(buf.name, buf.size);
        })
        .def("destroy", &ExternalBuffer::destroy)
        .def_prop_ro("handle", [] (ExternalBuffer& buffer) { return (u64)buffer.handle; })
    ;

    nb::class_<Image, GfxObject>(m, "Image")
        .def(nb::init<nb::ref<Context>, u32, u32, VkFormat, VkImageUsageFlagBits, gfx::AllocPresets::Type, int, std::optional<nb::str>>(), nb::arg("ctx"), nb::arg("width"), nb::arg("height"), nb::arg("format"), nb::arg("usage_flags"), nb::arg("alloc_type"), nb::arg("samples") = 1, nb::arg("name") = nb::none())
        .def("__repr__", [](Image& image) {
            return nb::str("Image(name={}, width={}, height={}, format={}, samples={})").format(image.name, image.width, image.height, image.format, image.samples);
        })
        .def("destroy", &Image::destroy)
        .def_static("from_data", &Image::from_data,
            nb::arg("ctx"),
            nb::arg("data"),
            nb::arg("usage"),
            nb::arg("width"),
            nb::arg("height"),
            nb::arg("format"),
            nb::arg("usage_flags"),
            nb::arg("alloc_type"),
            nb::arg("samples") = 1,
            nb::arg("name") = nb::none()
        )
        .def_ro("width", &Image::width)
        .def_ro("height", &Image::height)
        .def_ro("format", &Image::format)
        .def_ro("samples", &Image::samples)
        .def_ro("alloc", &Image::alloc)
    ;

    nb::class_<Sampler, GfxObject>(m, "Sampler")
        .def(nb::init<
                nb::ref<Context>,
                VkFilter,
                VkFilter,
                VkSamplerMipmapMode,
                float,
                float,
                float,
                VkSamplerAddressMode,
                VkSamplerAddressMode,
                VkSamplerAddressMode,
                bool,
                float,
                bool,
                VkCompareOp,
                std::optional<nb::str>
            >(),
                nb::arg("ctx"),
                nb::arg("min_filter")        = VK_FILTER_NEAREST,
                nb::arg("mag_filter")        = VK_FILTER_NEAREST,
                nb::arg("mipmap_mode")       = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                nb::arg("mip_lod_bias")      = 0.0f,
                nb::arg("min_lod")           = 0.0f,
                nb::arg("max_lod")           = VK_LOD_CLAMP_NONE,
                nb::arg("u")                 = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                nb::arg("v")                 = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                nb::arg("w")                 = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                nb::arg("anisotroy_enabled") = false,
                nb::arg("max_anisotropy")    = 0.0f,
                nb::arg("compare_enable")    = false,
                nb::arg("compare_op")        = VK_COMPARE_OP_ALWAYS,
                nb::arg("name") = nb::none()
            )
        .def("__repr__", [](Sampler& sampler) {
            return nb::str("Sampler(name={})").format(sampler.name);
        })
        .def("destroy", &Sampler::destroy)
    ;

    nb::class_<AccelerationStructure, GfxObject>(m, "AccelerationStructure")
        .def(nb::init<nb::ref<Context>, const std::vector<AccelerationStructureMesh>, bool, std::optional<nb::str>>(), nb::arg("ctx"), nb::arg("meshes"), nb::arg("prefer_fast_build") = false, nb::arg("name") = nb::none())
        .def("__repr__", [](AccelerationStructure& as) {
            return nb::str("AccelerationStructure(name={})").format(as.name);
        })
        .def("destroy", &AccelerationStructure::destroy)
    ;

    nb::class_<Fence, GfxObject>(m, "Fence")
        .def(nb::init<nb::ref<Context>, bool, std::optional<nb::str>>(), nb::arg("ctx"), nb::arg("signaled") = false, nb::arg("name") = nb::none())
        .def("__repr__", [](Fence& fence) {
            return nb::str("Fence(name={})").format(fence.name);
        })
        .def("destroy", &Fence::destroy)
        .def("is_signaled", &Fence::is_signaled)
        .def("wait", &Fence::wait)
        .def("resest", &Fence::reset)
        .def("wait_and_reset", &Fence::wait_and_reset)
    ;

    nb::class_<Semaphore, GfxObject>(m, "Semaphore")
        .def(nb::init<nb::ref<Context>, std::optional<nb::str>>(), nb::arg("ctx"), nb::arg("name") = nb::none())
        .def("__repr__", [](Semaphore& semaphore) {
            return nb::str("Semaphore(name={})").format(semaphore.name);
        })
        .def("destroy", &Semaphore::destroy)
    ;

    nb::class_<TimelineSemaphore, Semaphore>(m, "TimelineSemaphore")
        .def(nb::init<nb::ref<Context>, u64, std::optional<nb::str>>(), nb::arg("ctx"), nb::arg("initial_value") = 0, nb::arg("name") = nb::none())
        .def("__repr__", [](TimelineSemaphore& semaphore) {
            return nb::str("TimelineSemaphore(name={})").format(semaphore.name);
        })
        .def("get_value", &TimelineSemaphore::get_value)
        .def("signal", &TimelineSemaphore::signal, nb::arg("value"))
        .def("wait", &TimelineSemaphore::wait, nb::arg("value"))
    ;

    nb::class_<ExternalSemaphore, Semaphore>(m, "ExternalSemaphore")
        .def(nb::init<nb::ref<Context>, std::optional<nb::str>>(), nb::arg("ctx"), nb::arg("name") = nb::none())
        .def("__repr__", [](ExternalSemaphore& semaphore) {
            return nb::str("ExternalSemaphore(name={}, handle={})").format(semaphore.name, (u64)semaphore.handle);
        })
        .def("destroy", &ExternalSemaphore::destroy)
        .def_prop_ro("handle", [] (ExternalSemaphore& semaphore) { return (u64)semaphore.handle; })
    ;

    nb::class_<ExternalTimelineSemaphore, TimelineSemaphore>(m, "ExternalTimelineSemaphore")
        .def(nb::init<nb::ref<Context>, u64, std::optional<nb::str>>(), nb::arg("ctx"), nb::arg("initial_value") = 0, nb::arg("name") = nb::none())
        .def("__repr__", [](ExternalTimelineSemaphore& semaphore) {
            return nb::str("ExternalTimelineSemaphore(name={}, handle={})").format(semaphore.name, (u64)semaphore.handle);
        })
        .def("destroy", &ExternalTimelineSemaphore::destroy)
        .def_prop_ro("handle", [] (ExternalTimelineSemaphore& semaphore) { return (u64)semaphore.handle; })
    ;

    nb::enum_<MemoryUsage>(m, "MemoryUsage")
        .value("NONE", MemoryUsage::None)
        .value("HOST_WRITE", MemoryUsage::HostWrite)
        .value("TRANSFER_SRC", MemoryUsage::TransferSrc)
        .value("TRANSFER_DST", MemoryUsage::TransferDst)
        .value("VERTEX_INPUT", MemoryUsage::VertexInput)
        .value("VERTEX_SHADER_UNIFORM", MemoryUsage::VertexShaderUniform)
        .value("GEOMETRY_SHADER_UNIFORM", MemoryUsage::GeometryShaderUniform)
        .value("FRAGMENT_SHADER_UNIFORM", MemoryUsage::FragmentShaderUniform)
        .value("COMPUTE_SHADER_UNIFORM", MemoryUsage::ComputeShaderUniform)
        .value("ANY_SHADER_UNIFORM", MemoryUsage::AnyShaderUniform)
        .value("IMAGE", MemoryUsage::Image)
        .value("IMAGE_READ_ONLY", MemoryUsage::ImageReadOnly)
        .value("IMAGE_WRITE_ONLY", MemoryUsage::ImageWriteOnly)
        .value("SHADER_READ_ONLY", MemoryUsage::ShaderReadOnly)
        .value("COLOR_ATTACHMENT", MemoryUsage::ColorAttachment)
        .value("COLOR_ATTACHMENT_WRITE_ONLY", MemoryUsage::ColorAttachmentWriteOnly)
        .value("DEPTH_STENCIL_ATTACHMENT", MemoryUsage::DepthStencilAttachment)
        .value("DEPTH_STENCIL_ATTACHMENT_READ_ONLY", MemoryUsage::DepthStencilAttachmentReadOnly)
        .value("DEPTH_STENCIL_ATTACHMENT_WRITE_ONLY", MemoryUsage::DepthStencilAttachmentWriteOnly)
        .value("PRESENT", MemoryUsage::Present)
        .value("ALL", MemoryUsage::All)
    ;

    nb::enum_<VkImageLayout>(m, "ImageLayout")
        .value("UNDEFINED"                                     , VK_IMAGE_LAYOUT_UNDEFINED)
        .value("GENERAL"                                       , VK_IMAGE_LAYOUT_GENERAL)
        .value("COLOR_ATTACHMENT_OPTIMAL"                      , VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
        .value("DEPTH_STENCIL_ATTACHMENT_OPTIMAL"              , VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .value("DEPTH_STENCIL_READ_ONLY_OPTIMAL"               , VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)
        .value("SHADER_READ_ONLY_OPTIMAL"                      , VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        .value("TRANSFER_SRC_OPTIMAL"                          , VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
        .value("TRANSFER_DST_OPTIMAL"                          , VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        .value("PREINITIALIZED"                                , VK_IMAGE_LAYOUT_PREINITIALIZED)
        .value("DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL"    , VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL)
        .value("DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL"    , VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL)
        .value("DEPTH_ATTACHMENT_OPTIMAL"                      , VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL)
        .value("DEPTH_READ_ONLY_OPTIMAL"                       , VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL)
        .value("STENCIL_ATTACHMENT_OPTIMAL"                    , VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL)
        .value("STENCIL_READ_ONLY_OPTIMAL"                     , VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL)
        .value("READ_ONLY_OPTIMAL"                             , VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL)
        .value("ATTACHMENT_OPTIMAL"                            , VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL)
        .value("RENDERING_LOCAL_READ"                          , VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ)
        .value("PRESENT_SRC"                                   , VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
        .value("VIDEO_DECODE_DST"                              , VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR)
        .value("VIDEO_DECODE_SRC"                              , VK_IMAGE_LAYOUT_VIDEO_DECODE_SRC_KHR)
        .value("VIDEO_DECODE_DPB"                              , VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR)
        .value("SHARED_PRESENT"                                , VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR)
        .value("FRAGMENT_DENSITY_MAP_OPTIMAL_EXT"              , VK_IMAGE_LAYOUT_FRAGMENT_DENSITY_MAP_OPTIMAL_EXT)
        .value("FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL"      , VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR)
        .value("VIDEO_ENCODE_DST"                              , VK_IMAGE_LAYOUT_VIDEO_ENCODE_DST_KHR)
        .value("VIDEO_ENCODE_SRC"                              , VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR)
        .value("VIDEO_ENCODE_DPB"                              , VK_IMAGE_LAYOUT_VIDEO_ENCODE_DPB_KHR)
        .value("ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT"          , VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT)
        .value("VIDEO_ENCODE_QUANTIZATION_MAP"                 , VK_IMAGE_LAYOUT_VIDEO_ENCODE_QUANTIZATION_MAP_KHR)
        .value("SHADING_RATE_OPTIMAL"                          , VK_IMAGE_LAYOUT_SHADING_RATE_OPTIMAL_NV)
    ;

    nb::enum_<VkResolveModeFlagBits>(m, "ResolveMode")
        .value("NONE",        VK_RESOLVE_MODE_NONE)
        .value("SAMPLE_ZERO", VK_RESOLVE_MODE_SAMPLE_ZERO_BIT)
        .value("AVERAGE",     VK_RESOLVE_MODE_AVERAGE_BIT)
        .value("MIN",         VK_RESOLVE_MODE_MIN_BIT)
        .value("MAX",         VK_RESOLVE_MODE_MAX_BIT)
    ;

    nb::enum_<VkPipelineStageFlagBits>(m, "PipelineStageFlags", nb::is_flag(), nb::is_arithmetic())
        .value("TOP_OF_PIPE"                          , VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT)
        .value("DRAW_INDIRECT"                        , VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT)
        .value("VERTEX_INPUT"                         , VK_PIPELINE_STAGE_VERTEX_INPUT_BIT)
        .value("VERTEX_SHADER"                        , VK_PIPELINE_STAGE_VERTEX_SHADER_BIT)
        .value("TESSELLATION_CONTROL_SHADER"          , VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT)
        .value("TESSELLATION_EVALUATION_SHADER"       , VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT)
        .value("GEOMETRY_SHADER"                      , VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT)
        .value("FRAGMENT_SHADER"                      , VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
        .value("EARLY_FRAGMENT_TESTS"                 , VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT)
        .value("LATE_FRAGMENT_TESTS"                  , VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT)
        .value("COLOR_ATTACHMENT_OUTPUT"              , VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)
        .value("COMPUTE_SHADER"                       , VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
        .value("TRANSFER"                             , VK_PIPELINE_STAGE_TRANSFER_BIT)
        .value("BOTTOM_OF_PIPE"                       , VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT)
        .value("HOST"                                 , VK_PIPELINE_STAGE_HOST_BIT)
        .value("ALL_GRAPHICS"                         , VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT)
        .value("ALL_COMMANDS"                         , VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)
        .value("TRANSFORM_FEEDBACK"                   , VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT)
        .value("CONDITIONAL_RENDERING"                , VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT)
        .value("ACCELERATION_STRUCTURE_BUILD"         , VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR)
        .value("RAY_TRACING_SHADER"                   , VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR)
        .value("FRAGMENT_DENSITY_PROCESS"             , VK_PIPELINE_STAGE_FRAGMENT_DENSITY_PROCESS_BIT_EXT)
        .value("FRAGMENT_SHADING_RATE_ATTACHMENT"     , VK_PIPELINE_STAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR)
        .value("TASK_SHADER"                          , VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT)
        .value("MESH_SHADER"                          , VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT)
    ;

    nb::enum_<VkAttachmentLoadOp>(m, "LoadOp")
        .value("LOAD"     , VK_ATTACHMENT_LOAD_OP_LOAD)
        .value("CLEAR"    , VK_ATTACHMENT_LOAD_OP_CLEAR)
        .value("DONT_CARE", VK_ATTACHMENT_LOAD_OP_DONT_CARE)
    ;

    nb::enum_<VkAttachmentStoreOp>(m, "StoreOp")
        .value("STORE"    , VK_ATTACHMENT_STORE_OP_STORE)
        .value("DONT_CARE", VK_ATTACHMENT_STORE_OP_DONT_CARE)
        .value("NONE"     , VK_ATTACHMENT_STORE_OP_NONE)
    ;

    nb::class_<RenderingAttachment>(m, "RenderingAttachment",
        nb::intrusive_ptr<RenderingAttachment>([](RenderingAttachment *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<nb::ref<Image>, VkAttachmentLoadOp, VkAttachmentStoreOp, std::array<float, 4>, std::optional<nb::ref<Image>>, VkResolveModeFlagBits>(),
            nb::arg("image"), nb::arg("load_op") = VK_ATTACHMENT_LOAD_OP_LOAD, nb::arg("store_op") = VK_ATTACHMENT_STORE_OP_STORE, nb::arg("clear") = std::array<float,4>({0.0f, 0.0f, 0.0f, 0.0f}), nb::arg("resolve_image") = nb::none(), nb::arg("resolve_mode") = VK_RESOLVE_MODE_NONE)
    ;

    nb::class_<DepthAttachment>(m, "DepthAttachment",
        nb::intrusive_ptr<DepthAttachment>([](DepthAttachment *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<nb::ref<Image>, VkAttachmentLoadOp, VkAttachmentStoreOp, float>(),
            nb::arg("image"), nb::arg("load_op") = VK_ATTACHMENT_LOAD_OP_LOAD, nb::arg("store_op") = VK_ATTACHMENT_STORE_OP_STORE, nb::arg("clear") = 0.0f)
    ;

    nb::class_<CommandBuffer::RenderingManager>(m, "RenderingManager")
        .def("__enter__", &CommandBuffer::RenderingManager::enter)
        .def("__exit__", &CommandBuffer::RenderingManager::exit, nb::arg("exc_type").none(), nb::arg("exc_val").none(), nb::arg("exc_tb").none())
    ;

    nb::enum_<VkImageAspectFlagBits>(m, "ImageAspectFlags", nb::is_flag(), nb::is_arithmetic())
        .value("NONE",           VK_IMAGE_ASPECT_NONE)
        .value("COLOR",          VK_IMAGE_ASPECT_COLOR_BIT)
        .value("DEPTH",          VK_IMAGE_ASPECT_DEPTH_BIT)
        .value("STENCIL",        VK_IMAGE_ASPECT_STENCIL_BIT)
        .value("METADATA",       VK_IMAGE_ASPECT_METADATA_BIT)
        .value("PLANE_0",        VK_IMAGE_ASPECT_PLANE_0_BIT)
        .value("PLANE_1",        VK_IMAGE_ASPECT_PLANE_1_BIT)
        .value("PLANE_2",        VK_IMAGE_ASPECT_PLANE_2_BIT)
        .value("MEMORY_PLANE_0", VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT)
        .value("MEMORY_PLANE_1", VK_IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT)
        .value("MEMORY_PLANE_2", VK_IMAGE_ASPECT_MEMORY_PLANE_2_BIT_EXT)
        .value("MEMORY_PLANE_3", VK_IMAGE_ASPECT_MEMORY_PLANE_3_BIT_EXT)
    ;

    nb::enum_<VkIndexType>(m, "IndexType")
        .value("UINT16",    VK_INDEX_TYPE_UINT16)
        .value("UINT32",    VK_INDEX_TYPE_UINT32)
        .value("UINT8",     VK_INDEX_TYPE_UINT8)
        .value("NONE_KHR",  VK_INDEX_TYPE_NONE_KHR)
        .value("UINT8_KHR", VK_INDEX_TYPE_UINT8_KHR)
    ;

    nb::class_<CommandBuffer, GfxObject>(m, "CommandBuffer")
        .def(nb::init<nb::ref<Context>, std::optional<u32>, std::optional<nb::str>>(), nb::arg("ctx"), nb::arg("queue_family_index") = nb::none(), nb::arg("name") = nb::none())
        .def("__enter__", &CommandBuffer::enter)
        .def("__exit__", &CommandBuffer::exit, nb::arg("exc_type").none(), nb::arg("exc_val").none(), nb::arg("exc_tb").none())
        .def("__repr__", [](CommandBuffer& buf) {
            return nb::str("CommandBuffer(name={})").format(buf.name);
        })
        .def("destroy", &CommandBuffer::destroy)
        .def("begin", &CommandBuffer::begin)
        .def("end", &CommandBuffer::end)
        .def("memory_barrier", &CommandBuffer::memory_barrier, nb::arg("src"), nb::arg("dst"))
        .def("buffer_barrier", &CommandBuffer::buffer_barrier, nb::arg("buffer"), nb::arg("src"), nb::arg("dst"), nb::arg("src_queue_family_index") = VK_QUEUE_FAMILY_IGNORED, nb::arg("dst_queue_family_index") = VK_QUEUE_FAMILY_IGNORED)
        .def("image_barrier", &CommandBuffer::image_barrier, nb::arg("image"), nb::arg("dst_layout"), nb::arg("src_usage"), nb::arg("dst_usage"), nb::arg("src_queue_family_index") = VK_QUEUE_FAMILY_IGNORED, nb::arg("dst_queue_family_index") = VK_QUEUE_FAMILY_IGNORED, nb::arg("aspect_mask") = VK_IMAGE_ASPECT_COLOR_BIT, nb::arg("undefined") = false)
        .def("begin_rendering", &CommandBuffer::begin_rendering, nb::arg("render_area"), nb::arg("color_attachments"), nb::arg("depth") = nb::none())
        .def("end_rendering", &CommandBuffer::end_rendering)
        .def("rendering", &CommandBuffer::rendering, nb::arg("render_area"), nb::arg("color_attachments"), nb::arg("depth") = nb::none())
        .def("set_viewport", &CommandBuffer::set_viewport, nb::arg("viewport"))
        .def("set_scissors", &CommandBuffer::set_scissors, nb::arg("scissors"))
        .def("bind_pipeline", &CommandBuffer::bind_pipeline,
            nb::arg("pipeline")
        )
        .def("bind_descriptor_sets", &CommandBuffer::bind_descriptor_sets,
            nb::arg("pipeline"),
            nb::arg("descriptor_sets"),
            nb::arg("dynamic_offsets") = std::vector<u32>(),
            nb::arg("first_descriptor_set") = 0
        )
        .def("push_constants", &CommandBuffer::push_constants,
            nb::arg("pipeline"),
            nb::arg("push_constants"),
            nb::arg("offset") = 0
        )
        .def("bind_compute_pipeline", &CommandBuffer::bind_compute_pipeline,
            nb::arg("pipeline"),
            nb::arg("descriptor_sets") = std::vector<nb::ref<DescriptorSet>>(),
            nb::arg("dynamic_offsets") = std::vector<u32>(),
            nb::arg("first_descriptor_set") = 0,
            nb::arg("push_constants") = std::optional<nb::bytes>(),
            nb::arg("push_constants_offset") = 0
        )
        .def("bind_vertex_buffers", &CommandBuffer::bind_vertex_buffers,
            nb::arg("vertex_buffers"),
            nb::arg("first_vertex_buffer_binding") = 0
        )
        .def("bind_index_buffers", &CommandBuffer::bind_index_buffer,
            nb::arg("index_buffer"),
            nb::arg("index_buffer_offset") = 0,
            nb::arg("index_type") = VK_INDEX_TYPE_UINT32
        )
        .def("bind_graphics_pipeline", &CommandBuffer::bind_graphics_pipeline,
            nb::arg("pipeline"),
            nb::arg("descriptor_sets") = std::vector<nb::ref<DescriptorSet>>(),
            nb::arg("dynamic_offsets") = std::vector<u32>(),
            nb::arg("first_descriptor_set") = 0,
            nb::arg("push_constants") = std::optional<nb::bytes>(),
            nb::arg("push_constants_offset") = 0,
            nb::arg("vertex_buffers") = std::vector<nb::ref<Buffer>>(),
            nb::arg("first_vertex_buffer_binding") = 0,
            nb::arg("index_buffer") = std::optional<nb::ref<Buffer>>(),
            nb::arg("index_buffer_offset") = 0,
            nb::arg("index_type") = VK_INDEX_TYPE_UINT32
        )
        .def("dispatch", &CommandBuffer::dispatch,
            nb::arg("groups_x"),
            nb::arg("groups_y") = 1,
            nb::arg("groups_z") = 1
        )
        .def("draw", &CommandBuffer::draw,
            nb::arg("num_vertices"),
            nb::arg("num_instances") = 1,
            nb::arg("first_vertex") = 0,
            nb::arg("first_instance") = 0
        )
        .def("draw_indexed", &CommandBuffer::draw_indexed,
            nb::arg("num_indices"),
            nb::arg("num_instances") = 1,
            nb::arg("first_index") = 0,
            nb::arg("vertex_offset") = 0,
            nb::arg("first_instance") = 0
        )
        .def("copy_buffer", &CommandBuffer::copy_buffer,
            nb::arg("src"),
            nb::arg("dst")
        )
        .def("copy_buffer_range", &CommandBuffer::copy_buffer_range,
            nb::arg("src"),
            nb::arg("dst"),
            nb::arg("size"),
            nb::arg("src_offset") = 0,
            nb::arg("dst_offset") = 0
        )
        .def("copy_image_to_buffer", &CommandBuffer::copy_image_to_buffer,
            nb::arg("image"),
            nb::arg("buffer"),
            nb::arg("buffer_offset_in_bytes") = 0
        )
        .def("copy_buffer_to_image", &CommandBuffer::copy_buffer_to_image,
            nb::arg("buffer"),
            nb::arg("image"),
            nb::arg("buffer_offset_in_bytes") = 0
        )
        .def("copy_buffer_to_image_range", &CommandBuffer::copy_buffer_to_image_range,
            nb::arg("buffer"),
            nb::arg("image"),
            nb::arg("image_width"),
            nb::arg("image_height"),
            nb::arg("image_x") = 0,
            nb::arg("image_y") = 0,
            nb::arg("buffer_offset_in_bytes") = 0,
            nb::arg("buffer_row_stride_in_texels") = 0
        )
        .def("clear_color_image", &CommandBuffer::clear_color_image,
            nb::arg("image"),
            nb::arg("color")
        )
        .def("clear_depth_stencil_image", &CommandBuffer::clear_depth_stencil_image,
            nb::arg("image"),
            nb::arg("depth") = nb::none(),
            nb::arg("stencil") = nb::none()
        )
        .def("blit_image", &CommandBuffer::blit_image,
            nb::arg("src"),
            nb::arg("dst"),
            nb::arg("filter") = VK_FILTER_NEAREST,
            nb::arg("src_aspect") = VK_IMAGE_ASPECT_COLOR_BIT,
            nb::arg("dst_aspect") = VK_IMAGE_ASPECT_COLOR_BIT
        )
        .def("blit_image_range", &CommandBuffer::blit_image_range,
            nb::arg("src"),
            nb::arg("src_width"),
            nb::arg("src_height"),
            nb::arg("dst"),
            nb::arg("dst_width"),
            nb::arg("dst_height"),
            nb::arg("filter") = VK_FILTER_NEAREST,
            nb::arg("src_x") = 0,
            nb::arg("src_y") = 0,
            nb::arg("src_aspect") = VK_IMAGE_ASPECT_COLOR_BIT,
            nb::arg("dst_x") = 0,
            nb::arg("dst_y") = 0,
            nb::arg("dst_aspect") = VK_IMAGE_ASPECT_COLOR_BIT
        )
        .def("resolve_image", &CommandBuffer::resolve_image,
            nb::arg("src"),
            nb::arg("dst"),
            nb::arg("src_aspect") = VK_IMAGE_ASPECT_COLOR_BIT,
            nb::arg("dst_aspect") = VK_IMAGE_ASPECT_COLOR_BIT
        )
        .def("resolve_image_range", &CommandBuffer::resolve_image_range,
            nb::arg("src"),
            nb::arg("dst"),
            nb::arg("width"),
            nb::arg("height"),
            nb::arg("src_x") = 0,
            nb::arg("src_y") = 0,
            nb::arg("src_aspect") = VK_IMAGE_ASPECT_COLOR_BIT,
            nb::arg("dst_x") = 0,
            nb::arg("dst_y") = 0,
            nb::arg("dst_aspect") = VK_IMAGE_ASPECT_COLOR_BIT
        )
        .def("reset_query_pool", &CommandBuffer::reset_query_pool,
            nb::arg("pool")
        )
        .def("write_timestamp", &CommandBuffer::write_timestamp,
            nb::arg("pool"),
            nb::arg("index"),
            nb::arg("stage")
        )
        .def("begin_label", &CommandBuffer::begin_label, nb::arg("name"), nb::arg("color") = nb::none())
        .def("end_label", &CommandBuffer::end_label)
        .def("insert_label", &CommandBuffer::insert_label, nb::arg("name"), nb::arg("color") = nb::none())
        .def("set_line_width", &CommandBuffer::set_line_width, nb::arg("width"))
    ;

    nb::class_<Shader, GfxObject>(m, "Shader")
        .def(nb::init<nb::ref<Context>, const nb::bytes&, std::optional<nb::str>>(), nb::arg("ctx"), nb::arg("code"), nb::arg("name") = nb::none())
        .def("__repr__", [](Shader& shader) {
            return nb::str("Shader(name={})").format(shader.name);
        })
        .def("destroy", &Shader::destroy)
    ;

    nb::enum_<VkShaderStageFlagBits>(m, "Stage", nb::is_flag(), nb::is_arithmetic())
        .value("VERTEX",                  VK_SHADER_STAGE_VERTEX_BIT)
        .value("TESSELLATION_CONTROL",    VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT)
        .value("TESSELLATION_EVALUATION", VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT)
        .value("GEOMETRY",                VK_SHADER_STAGE_GEOMETRY_BIT)
        .value("FRAGMENT",                VK_SHADER_STAGE_FRAGMENT_BIT)
        .value("COMPUTE",                 VK_SHADER_STAGE_COMPUTE_BIT)
        .value("RAYGEN",                  VK_SHADER_STAGE_RAYGEN_BIT_KHR)
        .value("ANY_HIT",                 VK_SHADER_STAGE_ANY_HIT_BIT_KHR)
        .value("CLOSEST_HIT",             VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
        .value("MISS",                    VK_SHADER_STAGE_MISS_BIT_KHR)
        .value("INTERSECTION",            VK_SHADER_STAGE_INTERSECTION_BIT_KHR)
        .value("CALLABLE",                VK_SHADER_STAGE_CALLABLE_BIT_KHR)
        .value("TASK_EXT",                VK_SHADER_STAGE_TASK_BIT_EXT)
        .value("MESH_EXT",                VK_SHADER_STAGE_MESH_BIT_EXT)
    ;

    nb::class_<PipelineStage>(m, "PipelineStage",
        nb::intrusive_ptr<PipelineStage>([](PipelineStage *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<nb::ref<Shader>, VkShaderStageFlagBits, std::string>(), nb::arg("shader"), nb::arg("stage"), nb::arg("entry") = "main")
    ;

    nb::enum_<VkVertexInputRate>(m, "VertexInputRate")
        .value("VERTEX", VK_VERTEX_INPUT_RATE_VERTEX)
        .value("INSTANCE", VK_VERTEX_INPUT_RATE_VERTEX)
    ;

    nb::class_<VertexBinding>(m, "VertexBinding")
        .def(nb::init<u32, u32, VkVertexInputRate>(), nb::arg("binding"), nb::arg("stride"), nb::arg("input_rate") = VK_VERTEX_INPUT_RATE_VERTEX)
    ;

    nb::class_<gfx::FormatInfo>(m, "FormatInfo")
        .def_ro("size", &gfx::FormatInfo::size)
        .def_ro("channels", &gfx::FormatInfo::channels)
        .def_ro("size_of_block_in_bytes", &gfx::FormatInfo::size_of_block_in_bytes)
        .def_ro("block_side_in_pixels", &gfx::FormatInfo::block_side_in_pixels)
        .def("__repr__", [](gfx::FormatInfo info) {
            return nb::str("FormatInfo(size={}, channels={}, size_of_blocks_in_bytes={}, block_side_in_pixels={})").format(info.size, info.channels, info.size_of_block_in_bytes, info.block_side_in_pixels);
        })
    ;

    m.def("get_format_info", gfx::GetFormatInfo);

    nb::enum_<VkFormat>(m, "Format")
        .value("UNDEFINED", VK_FORMAT_UNDEFINED)
        .value("R4G4_UNORM_PACK8", VK_FORMAT_R4G4_UNORM_PACK8)
        .value("R4G4B4A4_UNORM_PACK16", VK_FORMAT_R4G4B4A4_UNORM_PACK16)
        .value("B4G4R4A4_UNORM_PACK16", VK_FORMAT_B4G4R4A4_UNORM_PACK16)
        .value("R5G6B5_UNORM_PACK16", VK_FORMAT_R5G6B5_UNORM_PACK16)
        .value("B5G6R5_UNORM_PACK16", VK_FORMAT_B5G6R5_UNORM_PACK16)
        .value("R5G5B5A1_UNORM_PACK16", VK_FORMAT_R5G5B5A1_UNORM_PACK16)
        .value("B5G5R5A1_UNORM_PACK16", VK_FORMAT_B5G5R5A1_UNORM_PACK16)
        .value("A1R5G5B5_UNORM_PACK16", VK_FORMAT_A1R5G5B5_UNORM_PACK16)
        .value("R8_UNORM", VK_FORMAT_R8_UNORM)
        .value("R8_SNORM", VK_FORMAT_R8_SNORM)
        .value("R8_USCALED", VK_FORMAT_R8_USCALED)
        .value("R8_SSCALED", VK_FORMAT_R8_SSCALED)
        .value("R8_UINT", VK_FORMAT_R8_UINT)
        .value("R8_SINT", VK_FORMAT_R8_SINT)
        .value("R8_SRGB", VK_FORMAT_R8_SRGB)
        .value("R8G8_UNORM", VK_FORMAT_R8G8_UNORM)
        .value("R8G8_SNORM", VK_FORMAT_R8G8_SNORM)
        .value("R8G8_USCALED", VK_FORMAT_R8G8_USCALED)
        .value("R8G8_SSCALED", VK_FORMAT_R8G8_SSCALED)
        .value("R8G8_UINT", VK_FORMAT_R8G8_UINT)
        .value("R8G8_SINT", VK_FORMAT_R8G8_SINT)
        .value("R8G8_SRGB", VK_FORMAT_R8G8_SRGB)
        .value("R8G8B8_UNORM", VK_FORMAT_R8G8B8_UNORM)
        .value("R8G8B8_SNORM", VK_FORMAT_R8G8B8_SNORM)
        .value("R8G8B8_USCALED", VK_FORMAT_R8G8B8_USCALED)
        .value("R8G8B8_SSCALED", VK_FORMAT_R8G8B8_SSCALED)
        .value("R8G8B8_UINT", VK_FORMAT_R8G8B8_UINT)
        .value("R8G8B8_SINT", VK_FORMAT_R8G8B8_SINT)
        .value("R8G8B8_SRGB", VK_FORMAT_R8G8B8_SRGB)
        .value("B8G8R8_UNORM", VK_FORMAT_B8G8R8_UNORM)
        .value("B8G8R8_SNORM", VK_FORMAT_B8G8R8_SNORM)
        .value("B8G8R8_USCALED", VK_FORMAT_B8G8R8_USCALED)
        .value("B8G8R8_SSCALED", VK_FORMAT_B8G8R8_SSCALED)
        .value("B8G8R8_UINT", VK_FORMAT_B8G8R8_UINT)
        .value("B8G8R8_SINT", VK_FORMAT_B8G8R8_SINT)
        .value("B8G8R8_SRGB", VK_FORMAT_B8G8R8_SRGB)
        .value("R8G8B8A8_UNORM", VK_FORMAT_R8G8B8A8_UNORM)
        .value("R8G8B8A8_SNORM", VK_FORMAT_R8G8B8A8_SNORM)
        .value("R8G8B8A8_USCALED", VK_FORMAT_R8G8B8A8_USCALED)
        .value("R8G8B8A8_SSCALED", VK_FORMAT_R8G8B8A8_SSCALED)
        .value("R8G8B8A8_UINT", VK_FORMAT_R8G8B8A8_UINT)
        .value("R8G8B8A8_SINT", VK_FORMAT_R8G8B8A8_SINT)
        .value("R8G8B8A8_SRGB", VK_FORMAT_R8G8B8A8_SRGB)
        .value("B8G8R8A8_UNORM", VK_FORMAT_B8G8R8A8_UNORM)
        .value("B8G8R8A8_SNORM", VK_FORMAT_B8G8R8A8_SNORM)
        .value("B8G8R8A8_USCALED", VK_FORMAT_B8G8R8A8_USCALED)
        .value("B8G8R8A8_SSCALED", VK_FORMAT_B8G8R8A8_SSCALED)
        .value("B8G8R8A8_UINT", VK_FORMAT_B8G8R8A8_UINT)
        .value("B8G8R8A8_SINT", VK_FORMAT_B8G8R8A8_SINT)
        .value("B8G8R8A8_SRGB", VK_FORMAT_B8G8R8A8_SRGB)
        .value("A8B8G8R8_UNORM_PACK32", VK_FORMAT_A8B8G8R8_UNORM_PACK32)
        .value("A8B8G8R8_SNORM_PACK32", VK_FORMAT_A8B8G8R8_SNORM_PACK32)
        .value("A8B8G8R8_USCALED_PACK32", VK_FORMAT_A8B8G8R8_USCALED_PACK32)
        .value("A8B8G8R8_SSCALED_PACK32", VK_FORMAT_A8B8G8R8_SSCALED_PACK32)
        .value("A8B8G8R8_UINT_PACK32", VK_FORMAT_A8B8G8R8_UINT_PACK32)
        .value("A8B8G8R8_SINT_PACK32", VK_FORMAT_A8B8G8R8_SINT_PACK32)
        .value("A8B8G8R8_SRGB_PACK32", VK_FORMAT_A8B8G8R8_SRGB_PACK32)
        .value("A2R10G10B10_UNORM_PACK32", VK_FORMAT_A2R10G10B10_UNORM_PACK32)
        .value("A2R10G10B10_SNORM_PACK32", VK_FORMAT_A2R10G10B10_SNORM_PACK32)
        .value("A2R10G10B10_USCALED_PACK32", VK_FORMAT_A2R10G10B10_USCALED_PACK32)
        .value("A2R10G10B10_SSCALED_PACK32", VK_FORMAT_A2R10G10B10_SSCALED_PACK32)
        .value("A2R10G10B10_UINT_PACK32", VK_FORMAT_A2R10G10B10_UINT_PACK32)
        .value("A2R10G10B10_SINT_PACK32", VK_FORMAT_A2R10G10B10_SINT_PACK32)
        .value("A2B10G10R10_UNORM_PACK32", VK_FORMAT_A2B10G10R10_UNORM_PACK32)
        .value("A2B10G10R10_SNORM_PACK32", VK_FORMAT_A2B10G10R10_SNORM_PACK32)
        .value("A2B10G10R10_USCALED_PACK32", VK_FORMAT_A2B10G10R10_USCALED_PACK32)
        .value("A2B10G10R10_SSCALED_PACK32", VK_FORMAT_A2B10G10R10_SSCALED_PACK32)
        .value("A2B10G10R10_UINT_PACK32", VK_FORMAT_A2B10G10R10_UINT_PACK32)
        .value("A2B10G10R10_SINT_PACK32", VK_FORMAT_A2B10G10R10_SINT_PACK32)
        .value("R16_UNORM", VK_FORMAT_R16_UNORM)
        .value("R16_SNORM", VK_FORMAT_R16_SNORM)
        .value("R16_USCALED", VK_FORMAT_R16_USCALED)
        .value("R16_SSCALED", VK_FORMAT_R16_SSCALED)
        .value("R16_UINT", VK_FORMAT_R16_UINT)
        .value("R16_SINT", VK_FORMAT_R16_SINT)
        .value("R16_SFLOAT", VK_FORMAT_R16_SFLOAT)
        .value("R16G16_UNORM", VK_FORMAT_R16G16_UNORM)
        .value("R16G16_SNORM", VK_FORMAT_R16G16_SNORM)
        .value("R16G16_USCALED", VK_FORMAT_R16G16_USCALED)
        .value("R16G16_SSCALED", VK_FORMAT_R16G16_SSCALED)
        .value("R16G16_UINT", VK_FORMAT_R16G16_UINT)
        .value("R16G16_SINT", VK_FORMAT_R16G16_SINT)
        .value("R16G16_SFLOAT", VK_FORMAT_R16G16_SFLOAT)
        .value("R16G16B16_UNORM", VK_FORMAT_R16G16B16_UNORM)
        .value("R16G16B16_SNORM", VK_FORMAT_R16G16B16_SNORM)
        .value("R16G16B16_USCALED", VK_FORMAT_R16G16B16_USCALED)
        .value("R16G16B16_SSCALED", VK_FORMAT_R16G16B16_SSCALED)
        .value("R16G16B16_UINT", VK_FORMAT_R16G16B16_UINT)
        .value("R16G16B16_SINT", VK_FORMAT_R16G16B16_SINT)
        .value("R16G16B16_SFLOAT", VK_FORMAT_R16G16B16_SFLOAT)
        .value("R16G16B16A16_UNORM", VK_FORMAT_R16G16B16A16_UNORM)
        .value("R16G16B16A16_SNORM", VK_FORMAT_R16G16B16A16_SNORM)
        .value("R16G16B16A16_USCALED", VK_FORMAT_R16G16B16A16_USCALED)
        .value("R16G16B16A16_SSCALED", VK_FORMAT_R16G16B16A16_SSCALED)
        .value("R16G16B16A16_UINT", VK_FORMAT_R16G16B16A16_UINT)
        .value("R16G16B16A16_SINT", VK_FORMAT_R16G16B16A16_SINT)
        .value("R16G16B16A16_SFLOAT", VK_FORMAT_R16G16B16A16_SFLOAT)
        .value("R32_UINT", VK_FORMAT_R32_UINT)
        .value("R32_SINT", VK_FORMAT_R32_SINT)
        .value("R32_SFLOAT", VK_FORMAT_R32_SFLOAT)
        .value("R32G32_UINT", VK_FORMAT_R32G32_UINT)
        .value("R32G32_SINT", VK_FORMAT_R32G32_SINT)
        .value("R32G32_SFLOAT", VK_FORMAT_R32G32_SFLOAT)
        .value("R32G32B32_UINT", VK_FORMAT_R32G32B32_UINT)
        .value("R32G32B32_SINT", VK_FORMAT_R32G32B32_SINT)
        .value("R32G32B32_SFLOAT", VK_FORMAT_R32G32B32_SFLOAT)
        .value("R32G32B32A32_UINT", VK_FORMAT_R32G32B32A32_UINT)
        .value("R32G32B32A32_SINT", VK_FORMAT_R32G32B32A32_SINT)
        .value("R32G32B32A32_SFLOAT", VK_FORMAT_R32G32B32A32_SFLOAT)
        .value("R64_UINT", VK_FORMAT_R64_UINT)
        .value("R64_SINT", VK_FORMAT_R64_SINT)
        .value("R64_SFLOAT", VK_FORMAT_R64_SFLOAT)
        .value("R64G64_UINT", VK_FORMAT_R64G64_UINT)
        .value("R64G64_SINT", VK_FORMAT_R64G64_SINT)
        .value("R64G64_SFLOAT", VK_FORMAT_R64G64_SFLOAT)
        .value("R64G64B64_UINT", VK_FORMAT_R64G64B64_UINT)
        .value("R64G64B64_SINT", VK_FORMAT_R64G64B64_SINT)
        .value("R64G64B64_SFLOAT", VK_FORMAT_R64G64B64_SFLOAT)
        .value("R64G64B64A64_UINT", VK_FORMAT_R64G64B64A64_UINT)
        .value("R64G64B64A64_SINT", VK_FORMAT_R64G64B64A64_SINT)
        .value("R64G64B64A64_SFLOAT", VK_FORMAT_R64G64B64A64_SFLOAT)
        .value("B10G11R11_UFLOAT_PACK32", VK_FORMAT_B10G11R11_UFLOAT_PACK32)
        .value("E5B9G9R9_UFLOAT_PACK32", VK_FORMAT_E5B9G9R9_UFLOAT_PACK32)
        .value("D16_UNORM", VK_FORMAT_D16_UNORM)
        .value("X8_D24_UNORM_PACK32", VK_FORMAT_X8_D24_UNORM_PACK32)
        .value("D32_SFLOAT", VK_FORMAT_D32_SFLOAT)
        .value("S8_UINT", VK_FORMAT_S8_UINT)
        .value("D16_UNORM_S8_UINT", VK_FORMAT_D16_UNORM_S8_UINT)
        .value("D24_UNORM_S8_UINT", VK_FORMAT_D24_UNORM_S8_UINT)
        .value("D32_SFLOAT_S8_UINT", VK_FORMAT_D32_SFLOAT_S8_UINT)
        .value("BC1_RGB_UNORM_BLOCK", VK_FORMAT_BC1_RGB_UNORM_BLOCK)
        .value("BC1_RGB_SRGB_BLOCK", VK_FORMAT_BC1_RGB_SRGB_BLOCK)
        .value("BC1_RGBA_UNORM_BLOCK", VK_FORMAT_BC1_RGBA_UNORM_BLOCK)
        .value("BC1_RGBA_SRGB_BLOCK", VK_FORMAT_BC1_RGBA_SRGB_BLOCK)
        .value("BC2_UNORM_BLOCK", VK_FORMAT_BC2_UNORM_BLOCK)
        .value("BC2_SRGB_BLOCK", VK_FORMAT_BC2_SRGB_BLOCK)
        .value("BC3_UNORM_BLOCK", VK_FORMAT_BC3_UNORM_BLOCK)
        .value("BC3_SRGB_BLOCK", VK_FORMAT_BC3_SRGB_BLOCK)
        .value("BC4_UNORM_BLOCK", VK_FORMAT_BC4_UNORM_BLOCK)
        .value("BC4_SNORM_BLOCK", VK_FORMAT_BC4_SNORM_BLOCK)
        .value("BC5_UNORM_BLOCK", VK_FORMAT_BC5_UNORM_BLOCK)
        .value("BC5_SNORM_BLOCK", VK_FORMAT_BC5_SNORM_BLOCK)
        .value("BC6H_UFLOAT_BLOCK", VK_FORMAT_BC6H_UFLOAT_BLOCK)
        .value("BC6H_SFLOAT_BLOCK", VK_FORMAT_BC6H_SFLOAT_BLOCK)
        .value("BC7_UNORM_BLOCK", VK_FORMAT_BC7_UNORM_BLOCK)
        .value("BC7_SRGB_BLOCK", VK_FORMAT_BC7_SRGB_BLOCK)
        .value("ETC2_R8G8B8_UNORM_BLOCK", VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK)
        .value("ETC2_R8G8B8_SRGB_BLOCK", VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK)
        .value("ETC2_R8G8B8A1_UNORM_BLOCK", VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK)
        .value("ETC2_R8G8B8A1_SRGB_BLOCK", VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK)
        .value("ETC2_R8G8B8A8_UNORM_BLOCK", VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK)
        .value("ETC2_R8G8B8A8_SRGB_BLOCK", VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK)
        .value("EAC_R11_UNORM_BLOCK", VK_FORMAT_EAC_R11_UNORM_BLOCK)
        .value("EAC_R11_SNORM_BLOCK", VK_FORMAT_EAC_R11_SNORM_BLOCK)
        .value("EAC_R11G11_UNORM_BLOCK", VK_FORMAT_EAC_R11G11_UNORM_BLOCK)
        .value("EAC_R11G11_SNORM_BLOCK", VK_FORMAT_EAC_R11G11_SNORM_BLOCK)
        .value("ASTC_4x4_UNORM_BLOCK", VK_FORMAT_ASTC_4x4_UNORM_BLOCK)
        .value("ASTC_4x4_SRGB_BLOCK", VK_FORMAT_ASTC_4x4_SRGB_BLOCK)
        .value("ASTC_5x4_UNORM_BLOCK", VK_FORMAT_ASTC_5x4_UNORM_BLOCK)
        .value("ASTC_5x4_SRGB_BLOCK", VK_FORMAT_ASTC_5x4_SRGB_BLOCK)
        .value("ASTC_5x5_UNORM_BLOCK", VK_FORMAT_ASTC_5x5_UNORM_BLOCK)
        .value("ASTC_5x5_SRGB_BLOCK", VK_FORMAT_ASTC_5x5_SRGB_BLOCK)
        .value("ASTC_6x5_UNORM_BLOCK", VK_FORMAT_ASTC_6x5_UNORM_BLOCK)
        .value("ASTC_6x5_SRGB_BLOCK", VK_FORMAT_ASTC_6x5_SRGB_BLOCK)
        .value("ASTC_6x6_UNORM_BLOCK", VK_FORMAT_ASTC_6x6_UNORM_BLOCK)
        .value("ASTC_6x6_SRGB_BLOCK", VK_FORMAT_ASTC_6x6_SRGB_BLOCK)
        .value("ASTC_8x5_UNORM_BLOCK", VK_FORMAT_ASTC_8x5_UNORM_BLOCK)
        .value("ASTC_8x5_SRGB_BLOCK", VK_FORMAT_ASTC_8x5_SRGB_BLOCK)
        .value("ASTC_8x6_UNORM_BLOCK", VK_FORMAT_ASTC_8x6_UNORM_BLOCK)
        .value("ASTC_8x6_SRGB_BLOCK", VK_FORMAT_ASTC_8x6_SRGB_BLOCK)
        .value("ASTC_8x8_UNORM_BLOCK", VK_FORMAT_ASTC_8x8_UNORM_BLOCK)
        .value("ASTC_8x8_SRGB_BLOCK", VK_FORMAT_ASTC_8x8_SRGB_BLOCK)
        .value("ASTC_10x5_UNORM_BLOCK", VK_FORMAT_ASTC_10x5_UNORM_BLOCK)
        .value("ASTC_10x5_SRGB_BLOCK", VK_FORMAT_ASTC_10x5_SRGB_BLOCK)
        .value("ASTC_10x6_UNORM_BLOCK", VK_FORMAT_ASTC_10x6_UNORM_BLOCK)
        .value("ASTC_10x6_SRGB_BLOCK", VK_FORMAT_ASTC_10x6_SRGB_BLOCK)
        .value("ASTC_10x8_UNORM_BLOCK", VK_FORMAT_ASTC_10x8_UNORM_BLOCK)
        .value("ASTC_10x8_SRGB_BLOCK", VK_FORMAT_ASTC_10x8_SRGB_BLOCK)
        .value("ASTC_10x10_UNORM_BLOCK", VK_FORMAT_ASTC_10x10_UNORM_BLOCK)
        .value("ASTC_10x10_SRGB_BLOCK", VK_FORMAT_ASTC_10x10_SRGB_BLOCK)
        .value("ASTC_12x10_UNORM_BLOCK", VK_FORMAT_ASTC_12x10_UNORM_BLOCK)
        .value("ASTC_12x10_SRGB_BLOCK", VK_FORMAT_ASTC_12x10_SRGB_BLOCK)
        .value("ASTC_12x12_UNORM_BLOCK", VK_FORMAT_ASTC_12x12_UNORM_BLOCK)
        .value("ASTC_12x12_SRGB_BLOCK", VK_FORMAT_ASTC_12x12_SRGB_BLOCK)
        .value("G8B8G8R8_422_UNORM", VK_FORMAT_G8B8G8R8_422_UNORM)
        .value("B8G8R8G8_422_UNORM", VK_FORMAT_B8G8R8G8_422_UNORM)
        .value("G8_B8_R8_3PLANE_420_UNORM", VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM)
        .value("G8_B8R8_2PLANE_420_UNORM", VK_FORMAT_G8_B8R8_2PLANE_420_UNORM)
        .value("G8_B8_R8_3PLANE_422_UNORM", VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM)
        .value("G8_B8R8_2PLANE_422_UNORM", VK_FORMAT_G8_B8R8_2PLANE_422_UNORM)
        .value("G8_B8_R8_3PLANE_444_UNORM", VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM)
        .value("R10X6_UNORM_PACK16", VK_FORMAT_R10X6_UNORM_PACK16)
        .value("R10X6G10X6_UNORM_2PACK16", VK_FORMAT_R10X6G10X6_UNORM_2PACK16)
        .value("R10X6G10X6B10X6A10X6_UNORM_4PACK16", VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16)
        .value("G10X6B10X6G10X6R10X6_422_UNORM_4PACK16", VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16)
        .value("B10X6G10X6R10X6G10X6_422_UNORM_4PACK16", VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16)
        .value("G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16", VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16)
        .value("G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16", VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16)
        .value("G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16", VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16)
        .value("G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16", VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16)
        .value("G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16", VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16)
        .value("R12X4_UNORM_PACK16", VK_FORMAT_R12X4_UNORM_PACK16)
        .value("R12X4G12X4_UNORM_2PACK16", VK_FORMAT_R12X4G12X4_UNORM_2PACK16)
        .value("R12X4G12X4B12X4A12X4_UNORM_4PACK16", VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16)
        .value("G12X4B12X4G12X4R12X4_422_UNORM_4PACK16", VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16)
        .value("B12X4G12X4R12X4G12X4_422_UNORM_4PACK16", VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16)
        .value("G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16", VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16)
        .value("G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16", VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16)
        .value("G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16", VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16)
        .value("G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16", VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16)
        .value("G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16", VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16)
        .value("G16B16G16R16_422_UNORM", VK_FORMAT_G16B16G16R16_422_UNORM)
        .value("B16G16R16G16_422_UNORM", VK_FORMAT_B16G16R16G16_422_UNORM)
        .value("G16_B16_R16_3PLANE_420_UNORM", VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM)
        .value("G16_B16R16_2PLANE_420_UNORM", VK_FORMAT_G16_B16R16_2PLANE_420_UNORM)
        .value("G16_B16_R16_3PLANE_422_UNORM", VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM)
        .value("G16_B16R16_2PLANE_422_UNORM", VK_FORMAT_G16_B16R16_2PLANE_422_UNORM)
        .value("G16_B16_R16_3PLANE_444_UNORM", VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM)
        .value("G8_B8R8_2PLANE_444_UNORM", VK_FORMAT_G8_B8R8_2PLANE_444_UNORM)
        .value("G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16", VK_FORMAT_G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16)
        .value("G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16", VK_FORMAT_G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16)
        .value("G16_B16R16_2PLANE_444_UNORM", VK_FORMAT_G16_B16R16_2PLANE_444_UNORM)
        .value("A4R4G4B4_UNORM_PACK16", VK_FORMAT_A4R4G4B4_UNORM_PACK16)
        .value("A4B4G4R4_UNORM_PACK16", VK_FORMAT_A4B4G4R4_UNORM_PACK16)
        .value("ASTC_4x4_SFLOAT_BLOCK", VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK)
        .value("ASTC_5x4_SFLOAT_BLOCK", VK_FORMAT_ASTC_5x4_SFLOAT_BLOCK)
        .value("ASTC_5x5_SFLOAT_BLOCK", VK_FORMAT_ASTC_5x5_SFLOAT_BLOCK)
        .value("ASTC_6x5_SFLOAT_BLOCK", VK_FORMAT_ASTC_6x5_SFLOAT_BLOCK)
        .value("ASTC_6x6_SFLOAT_BLOCK", VK_FORMAT_ASTC_6x6_SFLOAT_BLOCK)
        .value("ASTC_8x5_SFLOAT_BLOCK", VK_FORMAT_ASTC_8x5_SFLOAT_BLOCK)
        .value("ASTC_8x6_SFLOAT_BLOCK", VK_FORMAT_ASTC_8x6_SFLOAT_BLOCK)
        .value("ASTC_8x8_SFLOAT_BLOCK", VK_FORMAT_ASTC_8x8_SFLOAT_BLOCK)
        .value("ASTC_10x5_SFLOAT_BLOCK", VK_FORMAT_ASTC_10x5_SFLOAT_BLOCK)
        .value("ASTC_10x6_SFLOAT_BLOCK", VK_FORMAT_ASTC_10x6_SFLOAT_BLOCK)
        .value("ASTC_10x8_SFLOAT_BLOCK", VK_FORMAT_ASTC_10x8_SFLOAT_BLOCK)
        .value("ASTC_10x10_SFLOAT_BLOCK", VK_FORMAT_ASTC_10x10_SFLOAT_BLOCK)
        .value("ASTC_12x10_SFLOAT_BLOCK", VK_FORMAT_ASTC_12x10_SFLOAT_BLOCK)
        .value("ASTC_12x12_SFLOAT_BLOCK", VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK)
        .value("PVRTC1_2BPP_UNORM_BLOCK_IMG", VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG)
        .value("PVRTC1_4BPP_UNORM_BLOCK_IMG", VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG)
        .value("PVRTC2_2BPP_UNORM_BLOCK_IMG", VK_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG)
        .value("PVRTC2_4BPP_UNORM_BLOCK_IMG", VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG)
        .value("PVRTC1_2BPP_SRGB_BLOCK_IMG", VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG)
        .value("PVRTC1_4BPP_SRGB_BLOCK_IMG", VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG)
        .value("PVRTC2_2BPP_SRGB_BLOCK_IMG", VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG)
        .value("PVRTC2_4BPP_SRGB_BLOCK_IMG", VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG)
        .value("R16G16_SFIXED5_NV", VK_FORMAT_R16G16_SFIXED5_NV)
        .value("A1B5G5R5_UNORM_PACK16_KHR", VK_FORMAT_A1B5G5R5_UNORM_PACK16_KHR)
        .value("A8_UNORM_KHR", VK_FORMAT_A8_UNORM_KHR)
        .value("ASTC_4x4_SFLOAT_BLOCK_EXT", VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK_EXT)
        .value("ASTC_5x4_SFLOAT_BLOCK_EXT", VK_FORMAT_ASTC_5x4_SFLOAT_BLOCK_EXT)
        .value("ASTC_5x5_SFLOAT_BLOCK_EXT", VK_FORMAT_ASTC_5x5_SFLOAT_BLOCK_EXT)
        .value("ASTC_6x5_SFLOAT_BLOCK_EXT", VK_FORMAT_ASTC_6x5_SFLOAT_BLOCK_EXT)
        .value("ASTC_6x6_SFLOAT_BLOCK_EXT", VK_FORMAT_ASTC_6x6_SFLOAT_BLOCK_EXT)
        .value("ASTC_8x5_SFLOAT_BLOCK_EXT", VK_FORMAT_ASTC_8x5_SFLOAT_BLOCK_EXT)
        .value("ASTC_8x6_SFLOAT_BLOCK_EXT", VK_FORMAT_ASTC_8x6_SFLOAT_BLOCK_EXT)
        .value("ASTC_8x8_SFLOAT_BLOCK_EXT", VK_FORMAT_ASTC_8x8_SFLOAT_BLOCK_EXT)
        .value("ASTC_10x5_SFLOAT_BLOCK_EXT", VK_FORMAT_ASTC_10x5_SFLOAT_BLOCK_EXT)
        .value("ASTC_10x6_SFLOAT_BLOCK_EXT", VK_FORMAT_ASTC_10x6_SFLOAT_BLOCK_EXT)
        .value("ASTC_10x8_SFLOAT_BLOCK_EXT", VK_FORMAT_ASTC_10x8_SFLOAT_BLOCK_EXT)
        .value("ASTC_10x10_SFLOAT_BLOCK_EXT", VK_FORMAT_ASTC_10x10_SFLOAT_BLOCK_EXT)
        .value("ASTC_12x10_SFLOAT_BLOCK_EXT", VK_FORMAT_ASTC_12x10_SFLOAT_BLOCK_EXT)
        .value("ASTC_12x12_SFLOAT_BLOCK_EXT", VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK_EXT)
        .value("G8B8G8R8_422_UNORM_KHR", VK_FORMAT_G8B8G8R8_422_UNORM_KHR)
        .value("B8G8R8G8_422_UNORM_KHR", VK_FORMAT_B8G8R8G8_422_UNORM_KHR)
        .value("G8_B8_R8_3PLANE_420_UNORM_KHR", VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM_KHR)
        .value("G8_B8R8_2PLANE_420_UNORM_KHR", VK_FORMAT_G8_B8R8_2PLANE_420_UNORM_KHR)
        .value("G8_B8_R8_3PLANE_422_UNORM_KHR", VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM_KHR)
        .value("G8_B8R8_2PLANE_422_UNORM_KHR", VK_FORMAT_G8_B8R8_2PLANE_422_UNORM_KHR)
        .value("G8_B8_R8_3PLANE_444_UNORM_KHR", VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM_KHR)
        .value("R10X6_UNORM_PACK16_KHR", VK_FORMAT_R10X6_UNORM_PACK16_KHR)
        .value("R10X6G10X6_UNORM_2PACK16_KHR", VK_FORMAT_R10X6G10X6_UNORM_2PACK16_KHR)
        .value("R10X6G10X6B10X6A10X6_UNORM_4PACK16_KHR", VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16_KHR)
        .value("G10X6B10X6G10X6R10X6_422_UNORM_4PACK16_KHR", VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16_KHR)
        .value("B10X6G10X6R10X6G10X6_422_UNORM_4PACK16_KHR", VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16_KHR)
        .value("G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16_KHR", VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16_KHR)
        .value("G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16_KHR", VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16_KHR)
        .value("G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16_KHR", VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16_KHR)
        .value("G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16_KHR", VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16_KHR)
        .value("G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16_KHR", VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16_KHR)
        .value("R12X4_UNORM_PACK16_KHR", VK_FORMAT_R12X4_UNORM_PACK16_KHR)
        .value("R12X4G12X4_UNORM_2PACK16_KHR", VK_FORMAT_R12X4G12X4_UNORM_2PACK16_KHR)
        .value("R12X4G12X4B12X4A12X4_UNORM_4PACK16_KHR", VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16_KHR)
        .value("G12X4B12X4G12X4R12X4_422_UNORM_4PACK16_KHR", VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16_KHR)
        .value("B12X4G12X4R12X4G12X4_422_UNORM_4PACK16_KHR", VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16_KHR)
        .value("G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16_KHR", VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16_KHR)
        .value("G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16_KHR", VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16_KHR)
        .value("G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16_KHR", VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16_KHR)
        .value("G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16_KHR", VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16_KHR)
        .value("G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16_KHR", VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16_KHR)
        .value("G16B16G16R16_422_UNORM_KHR", VK_FORMAT_G16B16G16R16_422_UNORM_KHR)
        .value("B16G16R16G16_422_UNORM_KHR", VK_FORMAT_B16G16R16G16_422_UNORM_KHR)
        .value("G16_B16_R16_3PLANE_420_UNORM_KHR", VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM_KHR)
        .value("G16_B16R16_2PLANE_420_UNORM_KHR", VK_FORMAT_G16_B16R16_2PLANE_420_UNORM_KHR)
        .value("G16_B16_R16_3PLANE_422_UNORM_KHR", VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM_KHR)
        .value("G16_B16R16_2PLANE_422_UNORM_KHR", VK_FORMAT_G16_B16R16_2PLANE_422_UNORM_KHR)
        .value("G16_B16_R16_3PLANE_444_UNORM_KHR", VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM_KHR)
        .value("G8_B8R8_2PLANE_444_UNORM_EXT", VK_FORMAT_G8_B8R8_2PLANE_444_UNORM_EXT)
        .value("G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16_EXT", VK_FORMAT_G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16_EXT)
        .value("G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16_EXT", VK_FORMAT_G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16_EXT)
        .value("G16_B16R16_2PLANE_444_UNORM_EXT", VK_FORMAT_G16_B16R16_2PLANE_444_UNORM_EXT)
        .value("A4R4G4B4_UNORM_PACK16_EXT", VK_FORMAT_A4R4G4B4_UNORM_PACK16_EXT)
        .value("A4B4G4R4_UNORM_PACK16_EXT", VK_FORMAT_A4B4G4R4_UNORM_PACK16_EXT)
        .value("R16G16_S10_5_NV", VK_FORMAT_R16G16_S10_5_NV)
    ;

    nb::class_<VertexAttribute>(m, "VertexAttribute")
        .def(nb::init<u32, u32, VkFormat, u32>(), nb::arg("location"), nb::arg("binding"), nb::arg("format"), nb::arg("offset") = 0)
    ;

    nb::enum_<VkPrimitiveTopology>(m, "PrimitiveTopology")
        .value("POINT_LIST",                     VK_PRIMITIVE_TOPOLOGY_POINT_LIST)
        .value("LINE_LIST",                      VK_PRIMITIVE_TOPOLOGY_LINE_LIST)
        .value("LINE_STRIP",                     VK_PRIMITIVE_TOPOLOGY_LINE_STRIP)
        .value("TRIANGLE_LIST",                  VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .value("TRIANGLE_STRIP",                 VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP)
        .value("TRIANGLE_FAN",                   VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN)
        .value("LINE_LIST_WITH_ADJACENCY",       VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY)
        .value("LINE_STRIP_WITH_ADJACENCY",      VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY)
        .value("TRIANGLE_LIST_WITH_ADJACENCY",   VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY)
        .value("TRIANGLE_STRIP_WITH_ADJACENCY",  VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY)
        .value("PATCH_LIST",                     VK_PRIMITIVE_TOPOLOGY_PATCH_LIST)
    ;

    nb::class_<InputAssembly>(m, "InputAssembly")
        .def(nb::init<VkPrimitiveTopology, bool>(), nb::arg("primitive_topology") = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, nb::arg("primitive_restart_enable") = false);
    ;

    nb::enum_<VkPolygonMode>(m, "PolygonMode")
        .value("FILL",           VK_POLYGON_MODE_FILL)
        .value("LINE",           VK_POLYGON_MODE_LINE)
        .value("POINT",          VK_POLYGON_MODE_POINT)
        .value("FILL_RECTANGLE", VK_POLYGON_MODE_FILL_RECTANGLE_NV)
    ;

    nb::enum_<VkCullModeFlagBits>(m, "CullMode", nb::is_flag(), nb::is_arithmetic())
        .value("NONE",           VK_CULL_MODE_NONE)
        .value("FRONT",          VK_CULL_MODE_FRONT_BIT)
        .value("BACK",           VK_CULL_MODE_BACK_BIT)
        .value("FRONT_AND_BACK", VK_CULL_MODE_FRONT_AND_BACK)
    ;

    nb::enum_<VkFrontFace>(m, "FrontFace")
        .value("COUNTER_CLOCKWISE", VK_FRONT_FACE_COUNTER_CLOCKWISE)
        .value("CLOCKWISE",         VK_FRONT_FACE_CLOCKWISE)
    ;

    nb::class_<Rasterization>(m, "Rasterization")
        .def(nb::init<
                VkPolygonMode,
                VkCullModeFlagBits,
                VkFrontFace,
                bool,
                bool,
                bool,
                float
            >(),
            nb::arg("polygon_mode") = VK_POLYGON_MODE_FILL,
            nb::arg("cull_mode") = VK_CULL_MODE_NONE,
            nb::arg("front_face") = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            nb::arg("depth_bias_enable") = false,
            nb::arg("depth_clamp_enable") = false,
            nb::arg("dynamic_line_width") = false,
            nb::arg("line_width") = 1.0f)
    ;

    nb::enum_<VkDescriptorType>(m, "DescriptorType")
        .value("SAMPLER",                VK_DESCRIPTOR_TYPE_SAMPLER)
        .value("COMBINED_IMAGE_SAMPLER", VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
        .value("SAMPLED_IMAGE",          VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
        .value("STORAGE_IMAGE",          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
        .value("UNIFORM_TEXEL_BUFFER",   VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER)
        .value("STORAGE_TEXEL_BUFFER",   VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER)
        .value("UNIFORM_BUFFER",         VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
        .value("STORAGE_BUFFER",         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
        .value("UNIFORM_BUFFER_DYNAMIC", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC)
        .value("STORAGE_BUFFER_DYNAMIC", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
        .value("INPUT_ATTACHMENT",       VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT)
        .value("INLINE_UNIFORM_BLOCK",   VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK)
        .value("ACCELERATION_STRUCTURE", VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
        .value("SAMPLE_WEIGHT_IMAGE",    VK_DESCRIPTOR_TYPE_SAMPLE_WEIGHT_IMAGE_QCOM)
        .value("BLOCK_MATCH_IMAGE",      VK_DESCRIPTOR_TYPE_BLOCK_MATCH_IMAGE_QCOM)
        .value("MUTABLE",                VK_DESCRIPTOR_TYPE_MUTABLE_EXT)
    ;

    nb::class_<DescriptorSetEntry>(m, "DescriptorSetEntry")
        .def(nb::init<u32, VkDescriptorType>(), nb::arg("count"), nb::arg("type"))
    ;

    nb::enum_<VkDescriptorBindingFlagBits>(m, "DescriptorBindingFlags", nb::is_arithmetic() , nb::is_flag())
        .value("UPDATE_AFTER_BIND",           VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT)
        .value("UPDATE_UNUSED_WHILE_PENDING", VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT)
        .value("PARTIALLY_BOUND",             VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT)
        .value("VARIABLE_DESCRIPTOR_COUNT",   VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT)
    ;

    nb::class_<DescriptorSet, GfxObject>(m, "DescriptorSet")
        .def(nb::init<nb::ref<Context>, const std::vector<DescriptorSetEntry>&, VkDescriptorBindingFlagBits, std::optional<nb::str>>(), nb::arg("ctx"), nb::arg("entries"), nb::arg("flags") = VkDescriptorBindingFlagBits(), nb::arg("name") = nb::none())
        .def("__repr__", [](DescriptorSet& set) {
            return nb::str("DescriptorSet(name={})").format(set.name);
        })
        .def("destroy", &DescriptorSet::destroy)
        .def("write_buffer", &DescriptorSet::write_buffer, nb::arg("buffer"), nb::arg("type"), nb::arg("binding"), nb::arg("element") = 0, nb::arg("offset") = 0, nb::arg("size") = VK_WHOLE_SIZE)
        .def("write_image", &DescriptorSet::write_image, nb::arg("image"), nb::arg("layout"), nb::arg("type"), nb::arg("binding"), nb::arg("element") = 0)
        .def("write_sampler", &DescriptorSet::write_sampler, nb::arg("sampler"), nb::arg("binding"), nb::arg("element") = 0)
        .def("write_acceleration_structure", &DescriptorSet::write_acceleration_structure, nb::arg("acceleration_structure"), nb::arg("binding"), nb::arg("element") = 0)
    ;

    nb::enum_<VkBlendFactor>(m, "BlendFactor")
        .value("VK_BLEND_FACTOR_ZERO",                     VK_BLEND_FACTOR_ZERO)
        .value("VK_BLEND_FACTOR_ONE",                      VK_BLEND_FACTOR_ONE)
        .value("VK_BLEND_FACTOR_SRC_COLOR",                VK_BLEND_FACTOR_SRC_COLOR)
        .value("VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR",      VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR)
        .value("VK_BLEND_FACTOR_DST_COLOR",                VK_BLEND_FACTOR_DST_COLOR)
        .value("VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR",      VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR)
        .value("VK_BLEND_FACTOR_SRC_ALPHA",                VK_BLEND_FACTOR_SRC_ALPHA)
        .value("VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA",      VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA)
        .value("VK_BLEND_FACTOR_DST_ALPHA",                VK_BLEND_FACTOR_DST_ALPHA)
        .value("VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA",      VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA)
        .value("VK_BLEND_FACTOR_CONSTANT_COLOR",           VK_BLEND_FACTOR_CONSTANT_COLOR)
        .value("VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR", VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR)
        .value("VK_BLEND_FACTOR_CONSTANT_ALPHA",           VK_BLEND_FACTOR_CONSTANT_ALPHA)
        .value("VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA", VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA)
        .value("VK_BLEND_FACTOR_SRC_ALPHA_SATURATE",       VK_BLEND_FACTOR_SRC_ALPHA_SATURATE)
        .value("VK_BLEND_FACTOR_SRC1_COLOR",               VK_BLEND_FACTOR_SRC1_COLOR)
        .value("VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR",     VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR)
        .value("VK_BLEND_FACTOR_SRC1_ALPHA",               VK_BLEND_FACTOR_SRC1_ALPHA)
        .value("VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA",     VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA)
    ;

    nb::enum_<VkBlendOp>(m, "BlendOp")
        .value("OP_ADD",                VK_BLEND_OP_ADD)
        .value("OP_SUBTRACT",           VK_BLEND_OP_SUBTRACT)
        .value("OP_REVERSE_SUBTRACT",   VK_BLEND_OP_REVERSE_SUBTRACT)
        .value("OP_MIN",                VK_BLEND_OP_MIN)
        .value("OP_MAX",                VK_BLEND_OP_MAX)
        .value("OP_ZERO",               VK_BLEND_OP_ZERO_EXT)
        .value("OP_SRC",                VK_BLEND_OP_SRC_EXT)
        .value("OP_DST",                VK_BLEND_OP_DST_EXT)
        .value("OP_SRC_OVER",           VK_BLEND_OP_SRC_OVER_EXT)
        .value("OP_DST_OVER",           VK_BLEND_OP_DST_OVER_EXT)
        .value("OP_SRC_IN",             VK_BLEND_OP_SRC_IN_EXT)
        .value("OP_DST_IN",             VK_BLEND_OP_DST_IN_EXT)
        .value("OP_SRC_OUT",            VK_BLEND_OP_SRC_OUT_EXT)
        .value("OP_DST_OUT",            VK_BLEND_OP_DST_OUT_EXT)
        .value("OP_SRC_ATOP",           VK_BLEND_OP_SRC_ATOP_EXT)
        .value("OP_DST_ATOP",           VK_BLEND_OP_DST_ATOP_EXT)
        .value("OP_XOR",                VK_BLEND_OP_XOR_EXT)
        .value("OP_MULTIPLY",           VK_BLEND_OP_MULTIPLY_EXT)
        .value("OP_SCREEN",             VK_BLEND_OP_SCREEN_EXT)
        .value("OP_OVERLAY",            VK_BLEND_OP_OVERLAY_EXT)
        .value("OP_DARKEN",             VK_BLEND_OP_DARKEN_EXT)
        .value("OP_LIGHTEN",            VK_BLEND_OP_LIGHTEN_EXT)
        .value("OP_COLORDODGE",         VK_BLEND_OP_COLORDODGE_EXT)
        .value("OP_COLORBURN",          VK_BLEND_OP_COLORBURN_EXT)
        .value("OP_HARDLIGHT",          VK_BLEND_OP_HARDLIGHT_EXT)
        .value("OP_SOFTLIGHT",          VK_BLEND_OP_SOFTLIGHT_EXT)
        .value("OP_DIFFERENCE",         VK_BLEND_OP_DIFFERENCE_EXT)
        .value("OP_EXCLUSION",          VK_BLEND_OP_EXCLUSION_EXT)
        .value("OP_INVERT",             VK_BLEND_OP_INVERT_EXT)
        .value("OP_INVERT_RGB",         VK_BLEND_OP_INVERT_RGB_EXT)
        .value("OP_LINEARDODGE",        VK_BLEND_OP_LINEARDODGE_EXT)
        .value("OP_LINEARBURN",         VK_BLEND_OP_LINEARBURN_EXT)
        .value("OP_VIVIDLIGHT",         VK_BLEND_OP_VIVIDLIGHT_EXT)
        .value("OP_LINEARLIGHT",        VK_BLEND_OP_LINEARLIGHT_EXT)
        .value("OP_PINLIGHT",           VK_BLEND_OP_PINLIGHT_EXT)
        .value("OP_HARDMIX",            VK_BLEND_OP_HARDMIX_EXT)
        .value("OP_HSL_HUE",            VK_BLEND_OP_HSL_HUE_EXT)
        .value("OP_HSL_SATURATION",     VK_BLEND_OP_HSL_SATURATION_EXT)
        .value("OP_HSL_COLOR",          VK_BLEND_OP_HSL_COLOR_EXT)
        .value("OP_HSL_LUMINOSITY",     VK_BLEND_OP_HSL_LUMINOSITY_EXT)
        .value("OP_PLUS",               VK_BLEND_OP_PLUS_EXT)
        .value("OP_PLUS_CLAMPED",       VK_BLEND_OP_PLUS_CLAMPED_EXT)
        .value("OP_PLUS_CLAMPED_ALPHA", VK_BLEND_OP_PLUS_CLAMPED_ALPHA_EXT)
        .value("OP_PLUS_DARKER",        VK_BLEND_OP_PLUS_DARKER_EXT)
        .value("OP_MINUS",              VK_BLEND_OP_MINUS_EXT)
        .value("OP_MINUS_CLAMPED",      VK_BLEND_OP_MINUS_CLAMPED_EXT)
        .value("OP_CONTRAST",           VK_BLEND_OP_CONTRAST_EXT)
        .value("OP_INVERT_OVG",         VK_BLEND_OP_INVERT_OVG_EXT)
        .value("OP_RED",                VK_BLEND_OP_RED_EXT)
        .value("OP_GREEN",              VK_BLEND_OP_GREEN_EXT)
        .value("OP_BLUE",               VK_BLEND_OP_BLUE_EXT)
    ;

    nb::enum_<VkColorComponentFlagBits>(m, "ColorComponentFlags", nb::is_arithmetic() , nb::is_flag())
        .value("R", VK_COLOR_COMPONENT_R_BIT)
        .value("G", VK_COLOR_COMPONENT_G_BIT)
        .value("B", VK_COLOR_COMPONENT_B_BIT)
        .value("A", VK_COLOR_COMPONENT_A_BIT)
    ;

    nb::class_<Attachment>(m, "Attachment")
        .def(nb::init<
                VkFormat,
                bool,
                VkBlendFactor,
                VkBlendFactor,
                VkBlendOp,
                VkBlendFactor,
                VkBlendFactor,
                VkBlendOp,
                VkColorComponentFlags>(),

                nb::arg("format"),
                nb::arg("blend_enable")           = false,
                nb::arg("src_color_blend_factor") = VK_BLEND_FACTOR_ZERO,
                nb::arg("dst_color_blend_factor") = VK_BLEND_FACTOR_ZERO,
                nb::arg("color_blend_op")         = VK_BLEND_OP_ADD,
                nb::arg("src_alpha_blend_factor") = VK_BLEND_FACTOR_ZERO,
                nb::arg("dst_alpha_blend_factor") = VK_BLEND_FACTOR_ZERO,
                nb::arg("alpha_blend_op")         = VK_BLEND_OP_ADD,
                nb::arg("color_write_mask")       = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
            )
    ;

    nb::class_<PushConstantsRange>(m, "PushConstantsRange")
        .def(nb::init<u32, u32, VkShaderStageFlagBits>(), nb::arg("size"), nb::arg("offset") = 0, nb::arg("stages") = VK_SHADER_STAGE_ALL)
    ;

    nb::class_<AccelerationStructureMesh>(m, "AccelerationStructureMesh")
        .def(nb::init<
            VkDeviceAddress,
            u64,
            u32,
            VkFormat,
            VkDeviceAddress,
            VkIndexType,
            u32,
            std::array<float, 12>
        >(),
            nb::arg("vertices_address"),
            nb::arg("vertices_stride"),
            nb::arg("vertices_count"),
            nb::arg("vertices_format"),
            nb::arg("indices_address"),
            nb::arg("indices_type"),
            nb::arg("primitive_count"),
            nb::arg("transform")
        )
    ;

    nb::class_<Depth>(m, "Depth")
        .def(nb::init<VkFormat, bool, bool, VkCompareOp>(),
            nb::arg("format") = VK_FORMAT_UNDEFINED,
            nb::arg("test") = false,
            nb::arg("write") = false,
            nb::arg("op") = VK_COMPARE_OP_LESS
        )
    ;

    nb::class_<ComputePipeline, GfxObject>(m, "ComputePipeline")
        .def(nb::init<nb::ref<Context>,
                nb::ref<Shader>,
                nb::str,
                const std::vector<PushConstantsRange>&,
                const std::vector<nb::ref<DescriptorSet>>&,
                std::optional<nb::str>
            >(),
            nb::arg("ctx"),
            nb::arg("shader"),
            nb::arg("entry") = "main",
            nb::arg("push_constants_ranges") = std::vector<PushConstantsRange>(),
            nb::arg("descriptor_sets") = std::vector<nb::ref<DescriptorSet>>(),
            nb::arg("name") = nb::none()
        )
        .def("__repr__", [](ComputePipeline& pipeline) {
            return nb::str("ComputePipeline(name={})").format(pipeline.name);
        })
        .def("destroy", &ComputePipeline::destroy)
    ;
    nb::class_<GraphicsPipeline, GfxObject>(m, "GraphicsPipeline")
        .def(nb::init<nb::ref<Context>,
                const std::vector<nb::ref<PipelineStage>>&,
                const std::vector<VertexBinding>&,
                const std::vector<VertexAttribute>&,
                InputAssembly,
                Rasterization,
                const std::vector<PushConstantsRange>&,
                const std::vector<nb::ref<DescriptorSet>>&,
                u32,
                const std::vector<Attachment>&,
                Depth,
                std::optional<nb::str>
            >(),
            nb::arg("ctx"),
            nb::arg("stages") = std::vector<nb::ref<PipelineStage>>(),
            nb::arg("vertex_bindings") = std::vector<VertexBinding>(),
            nb::arg("vertex_attributes") = std::vector<VertexAttribute>(),
            nb::arg("input_assembly") = InputAssembly(),
            nb::arg("rasterization") = Rasterization(),
            nb::arg("push_constants_ranges") = std::vector<PushConstantsRange>(),
            nb::arg("descriptor_sets") = std::vector<nb::ref<DescriptorSet>>(),
            nb::arg("samples") = 1,
            nb::arg("attachments") = std::vector<Attachment>(),
            nb::arg("depth") = Depth(VK_FORMAT_UNDEFINED),
            nb::arg("name") = nb::none()
        )
        .def("__repr__", [](GraphicsPipeline& pipeline) {
            return nb::str("GraphicsPipeline(name={})").format(pipeline.name);
        })
        .def("destroy", &GraphicsPipeline::destroy)
    ;

    nb::enum_<gfx::SwapchainStatus>(m, "SwapchainStatus")
        .value("READY", gfx::SwapchainStatus::READY)
        .value("RESIZED", gfx::SwapchainStatus::RESIZED)
        .value("MINIMIZED", gfx::SwapchainStatus::MINIMIZED)
    ;

    m.def("process_events", [](bool wait) {
        if (wait) {
            nb::gil_scoped_release release_gil;
            gfx::ProcessEvents(true);
        } else {
            gfx::ProcessEvents(false);
        }
    }, nb::arg("wait"));
}
