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
#include <nanobind/ndarray.h>

#include <nanobind/intrusive/counter.h>
#include <nanobind/intrusive/ref.h>

#include <xpg/gui.h>

#include "function.h"

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

// TODO: likely can easily generalize to any N, T pairs, probably needs nested
// template, look at stl/array for reference.
template<>
struct type_caster<glm::ivec2> {
    NB_TYPE_CASTER(glm::ivec2, const_name(NB_TYPING_TUPLE "[int, int]"))

    using Caster = make_caster<int>;

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
        PyObject *temp;

        /* Will initialize 'temp' (NULL in the case of a failure.) */
        PyObject **o = seq_get_with_size(src.ptr(), 2, &temp);

        bool success = o != nullptr;

        Caster caster;
        flags = flags_for_local_caster<int>(flags);

        if (success) {
            for (size_t i = 0; i < 2; ++i) {
                if (!caster.from_python(o[i], flags, cleanup) ||
                    !caster.template can_cast<int>()) {
                    success = false;
                    break;
                }

                value[i] = caster.operator cast_t<int>();
            }

            Py_XDECREF(temp);
        }

        return success;
    }

    static handle from_cpp(glm::ivec2 &&src, rv_policy policy, cleanup_list *cleanup) {
        object ret = steal(PyTuple_New(2));

        if (ret.is_valid()) {
            Py_ssize_t index = 0;

            for (size_t i = 0; i < 2; ++i) {
                handle h = Caster::from_cpp(src[i], policy, cleanup);

                if (!h.is_valid()) {
                    ret.reset();
                    break;
                }

                NB_TUPLE_SET_ITEM(ret.ptr(), index++, h.ptr());
            }
        }

        return ret.release();
    }
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)

namespace nb = nanobind;

struct Context: public nb::intrusive_base {
    Context()
    {
        gfx::Result result;
        result = gfx::Init();
        if (result != gfx::Result::SUCCESS) {
            throw std::runtime_error("Failed to initialize platform");
        }

        Array<const char*> instance_extensions = gfx::GetPresentationInstanceExtensions();
        instance_extensions.add("VK_EXT_debug_report");

        Array<const char*> device_extensions;
        device_extensions.add("VK_KHR_swapchain");
        device_extensions.add("VK_KHR_dynamic_rendering");
#ifdef _WIN32
        device_extensions.add("VK_KHR_external_memory_win32");
        device_extensions.add("VK_KHR_external_semaphore_win32");
#else
        device_extensions.add("VK_KHR_external_memory_fd");
        device_extensions.add("VK_KHR_external_semaphore_fd");
#endif

        result = gfx::CreateContext(&vk, {
            .minimum_api_version = (u32)VK_API_VERSION_1_3,
            .instance_extensions = instance_extensions,
            .device_extensions = device_extensions,
            .device_features = gfx::DeviceFeatures::DYNAMIC_RENDERING | gfx::DeviceFeatures::DESCRIPTOR_INDEXING | gfx::DeviceFeatures::SYNCHRONIZATION_2 | gfx::DeviceFeatures::SCALAR_BLOCK_LAYOUT,
            .enable_validation_layer = true,
            // .enable_gpu_based_validation = true,
        });
        if (result != gfx::Result::SUCCESS) {
            throw std::runtime_error("Failed to initialize vulkan");
        }
    }

    ~Context() {
        gfx::WaitIdle(vk);
        gfx::DestroyContext(&vk);
        logging::info("gfx", "done");
    }

    gfx::Context vk;
};

struct GfxObject: public nb::intrusive_base {
    GfxObject() {}
    GfxObject(nb::ref<Context> ctx, bool owned)
        : ctx(ctx)
        , owned(owned)
    {}

    // Reference to main context
    nb::ref<Context> ctx;

    // If set the underlying object should be freed on destruction.
    // User created objects normally have this set to true,
    // context/swapchain owned objects have this set to false.
    bool owned = true;
};

struct Buffer: public GfxObject {
    Buffer(nb::ref<Context> ctx, usize size, VkBufferUsageFlags usage_flags, gfx::AllocPresets::Type alloc_type)
        : GfxObject(ctx, true)
    {
        VkResult vkr = gfx::CreateBuffer(&buffer, ctx->vk, size, {
            .usage = (VkBufferUsageFlags)usage_flags,
            .alloc = gfx::AllocPresets::Types[(size_t)alloc_type],
        });
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to create buffer");
        }
    }

    Buffer(nb::ref<Context> ctx)
        : GfxObject(ctx, true)
    { }

    ~Buffer() {
        destroy();
    }

    void destroy() {
        if(owned) {
            gfx::DestroyBuffer(&buffer, ctx->vk);
        }
    }

    static nb::ref<Buffer> from_data(nb::ref<Context> ctx, const nb::bytes& data, VkBufferUsageFlags usage_flags, gfx::AllocPresets::Type alloc_type) {
        std::unique_ptr<Buffer> self = std::make_unique<Buffer>(ctx);

        VkResult vkr = gfx::CreateBufferFromData(&self->buffer, ctx->vk, ArrayView<u8>((u8*)data.data(), data.size()), {
            .usage = (VkBufferUsageFlags)usage_flags,
            .alloc = gfx::AllocPresets::Types[(size_t)alloc_type],
        });
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to create buffer");
        }

        return self.release();
    }

    gfx::Buffer buffer = {};
};

struct Semaphore: public GfxObject {
    Semaphore(nb::ref<Context> ctx, bool external)
        : GfxObject(ctx, true)
    {
        VkResult vkr = gfx::CreateGPUSemaphore(ctx->vk.device, &semaphore, external);
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to create semaphore");
        }
    }
    Semaphore(nb::ref<Context> ctx): Semaphore(ctx, false) { }

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

struct ExternalSemaphore: public Semaphore {
    ExternalSemaphore(nb::ref<Context> ctx): Semaphore(ctx, true) {
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

struct ExternalBuffer: public Buffer {
    ExternalBuffer(nb::ref<Context> ctx, usize size, VkBufferUsageFlags usage_flags, gfx::AllocPresets::Type alloc_type)
        : Buffer(ctx)
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
            .pool = pool,
            .external = true,
        });
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to create buffer");
        }

        vkr = gfx::GetExternalHandleForBuffer(&handle, ctx->vk, buffer);
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to get external handle");
        }
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

    VmaPool pool;
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

enum class ImageUsage {
    None,
    Image,
    ImageReadOnly,
    ImageWriteOnly,
    ColorAttachment,
    ColorAttachmentWriteOnly,
    DepthStencilAttachment,
    DepthStencilAttachmentReadOnly,
    DepthStencilAttachmentWriteOnly,
    Present,
};

struct ImageUsageState {
    VkPipelineStageFlags2 first_stage;  // For execution barriers, first stage that r/w to this resource
    VkPipelineStageFlags2 last_stage;   // For execution barriers, last stage that r/w to this resource
    VkAccessFlags2 access;              // For memory bariers, all accesses to this resource
    VkImageLayout layout;               // Only used if underlying resource is an image
};

namespace ImageUsagePresets {
    constexpr ImageUsageState None {
        .first_stage = VK_PIPELINE_STAGE_2_NONE,
        .last_stage = VK_PIPELINE_STAGE_2_NONE,
        .access = 0,
        .layout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    constexpr ImageUsageState Image {
        .first_stage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .layout = VK_IMAGE_LAYOUT_GENERAL,
    };
    constexpr ImageUsageState ImageReadOnly {
        .first_stage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        .layout = VK_IMAGE_LAYOUT_GENERAL, // This can potentially be READ_ONLY or even SHADER_READ_ONLY?
    };
    constexpr ImageUsageState ImageWriteOnly {
        .first_stage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .layout = VK_IMAGE_LAYOUT_GENERAL, // This can potentially be READ_ONLY or even SHADER_READ_ONLY?
    };
    constexpr ImageUsageState ColorAttachment = {
        .first_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .access = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };
    constexpr ImageUsageState ColorAttachmentWriteOnly = {
        .first_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };
    constexpr ImageUsageState DepthStencilAttachment = {
        .first_stage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        .access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    constexpr ImageUsageState DepthStencilAttachmentReadOnly = {
        .first_stage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        .access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
        .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    constexpr ImageUsageState DepthStencilAttachmentWriteOnly = {
        .first_stage = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        .last_stage = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        .access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    constexpr ImageUsageState Present = {
        .first_stage = VK_PIPELINE_STAGE_2_NONE,
        .last_stage = VK_PIPELINE_STAGE_2_NONE,
        .access = 0,
        .layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
    };

    ImageUsageState Types[] = {
        None,
        Image,
        ImageReadOnly,
        ImageWriteOnly,
        ColorAttachment,
        ColorAttachmentWriteOnly,
        DepthStencilAttachment,
        DepthStencilAttachmentReadOnly,
        DepthStencilAttachmentWriteOnly,
        Present,
    };
}

struct Image: public GfxObject {
    Image(nb::ref<Context> ctx, u32 width, u32 height, VkFormat format, VkImageUsageFlags usage_flags, gfx::AllocPresets::Type alloc_type, int samples = 1)
        : GfxObject(ctx, true)
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
    }

    Image(nb::ref<Context> ctx, VkImage image, VkImageView view)
        : GfxObject(ctx, false) {
        this->image.image = image;
        this->image.view = view;
        this->image.allocation = 0;
    }

    Image(nb::ref<Context> ctx)
        : GfxObject(ctx, true)
    { }

    ~Image() {
        destroy();
    }

    void destroy() {
        if(owned) {
            gfx::DestroyImage(&image, ctx->vk);
        }
    }

    // static nb::ref<Image> from_data(nb::ref<Context> ctx, const nb::bytes& data, VkBufferUsageFlags usage_flags, gfx::AllocPresets::Type alloc_type) {
    //     std::unique_ptr<Buffer> self = std::make_unique<Buffer>(ctx);

    //     VkResult vkr = gfx::CreateBufferFromData(&self->buffer, ctx->vk, ArrayView<u8>((u8*)data.data(), data.size()), {
    //         .usage = (VkBufferUsageFlags)usage_flags,
    //         .alloc = gfx::AllocPresets::Types[(size_t)alloc_type],
    //     });
    //     if (vkr != VK_SUCCESS) {
    //         throw std::runtime_error("Failed to create buffer");
    //     }

    //     return self.release();
    // }

    gfx::Image image = {};
    ImageUsageState current_state = ImageUsagePresets::Types[(usize)ImageUsage::None];
};

struct Window;
struct GraphicsPipeline;
struct DescriptorSet;
struct Buffer;

struct RenderingAttachment {
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

struct DepthAttachment {
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

struct CommandBuffer: GfxObject {
    CommandBuffer(nb::ref<Context> ctx, VkCommandPool pool, VkCommandBuffer buffer, bool owned)
        : GfxObject(ctx, owned)
        , pool(pool)
        , buffer(buffer)
    {}

    void use_image(Image& image, ImageUsage usage) {
        assert((usize)usage < ArrayCount(ImageUsagePresets::Types));
        ImageUsageState new_state = ImageUsagePresets::Types[(usize)usage];

        // Rules:
        // - If layout transition: always need barrier
        // - If one of the uses is a write: memory barrier needed

        bool layout_transition = image.current_state.layout != new_state.layout;
        bool memory_barrier = has_any_write_access(image.current_state.access | new_state.access);
        if (layout_transition || memory_barrier) {
            // TODO: use a memory barrier if no layout transition?
            gfx::CmdImageBarrier(buffer, {
                .image = image.image.image,
                .src_stage = image.current_state.last_stage,
                .dst_stage = new_state.first_stage,
                .src_access = image.current_state.access,
                .dst_access = new_state.access,
                .new_layout = new_state.layout,
                .aspect_mask =
                    (VkImageAspectFlags)((
                        usage == ImageUsage::DepthStencilAttachment ||
                        usage == ImageUsage::DepthStencilAttachmentReadOnly ||
                        usage == ImageUsage::DepthStencilAttachmentWriteOnly
                    )? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT),
            });
        }

        image.current_state = new_state;
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
        RenderingManager(nb::ref<CommandBuffer> cmd, std::array<u32, 4> viewport, std::vector<RenderingAttachment> color, std::optional<DepthAttachment> depth)
            : cmd(cmd)
            , viewport(viewport)
            , color(std::move(color))
            , depth(depth)
        {}

        void enter() {
            cmd->begin_rendering(viewport, color, depth);
        }

        void exit(nb::object, nb::object, nb::object) {
            cmd->end_rendering();
        }

        nb::ref<CommandBuffer> cmd;
        std::array<u32, 4> viewport;
        std::vector<RenderingAttachment> color;
        std::optional<DepthAttachment> depth;
    };

    RenderingManager rendering(std::array<u32, 4> viewport, const std::vector<RenderingAttachment>& color, std::optional<DepthAttachment> depth) {
        return RenderingManager(this, viewport, std::move(color), depth);
    }

    void begin_rendering(std::array<u32, 4> viewport, const std::vector<RenderingAttachment>& color, std::optional<DepthAttachment> depth) {
        Array<gfx::RenderingAttachmentDesc> color_descs(color.size());
        for(usize i = 0; i < color_descs.length; i++) {
            VkClearColorValue clear;
            clear.float32[0] = color[i].clear[0];
            clear.float32[1] = color[i].clear[1];
            clear.float32[2] = color[i].clear[2];
            clear.float32[3] = color[i].clear[3];

            color_descs[i].view = color[i].image->image.view;
            color_descs[i].store_op = color[i].store_op;
            color_descs[i].load_op = color[i].load_op;
            color_descs[i].clear = clear;
            color_descs[i].resolve_mode = color[i].resolve_mode;
            color_descs[i].resolve_image_layout = color[i].resolve_image.has_value() ? color[i].resolve_image.value()->current_state.layout : VK_IMAGE_LAYOUT_UNDEFINED;
            color_descs[i].resolve_image_view = color[i].resolve_image.has_value() ? color[i].resolve_image.value()->image.view : VK_NULL_HANDLE;
        }

        gfx::DepthAttachmentDesc depth_desc = {};
        if(depth.has_value()) {
            depth_desc.view = depth->image->image.view;
            depth_desc.load_op = depth->load_op;
            depth_desc.store_op= depth->store_op;
            depth_desc.clear = depth->clear;
        }

        gfx::CmdBeginRendering(buffer, {
            .color = Span(color_descs),
            .depth = depth_desc,
            .offset_x = viewport[0],
            .offset_y = viewport[1],
            .width = viewport[2],
            .height = viewport[3],
        });
    }

    void end_rendering() {
        gfx::CmdEndRendering(buffer);
    }

    void bind_pipeline_state(
        const GraphicsPipeline& pipeline,
        const std::vector<nb::ref<DescriptorSet>> descriptor_sets,
        std::optional<nb::bytes> push_constants,
        const std::vector<nb::ref<Buffer>> vertex_buffers,
        std::optional<nb::ref<Buffer>> index_buffer,
        std::array<u32, 4> viewport,
        std::array<u32, 4> scissors
    );

    void draw_indexed(
        u32 num_indices,
        u32 num_instances,
        u32 first_index,
        s32 vertex_offset,
        u32 first_instance
    ) {
        vkCmdDrawIndexed(buffer, num_indices, num_instances, first_index, vertex_offset, first_instance);
    }

    VkCommandPool pool;
    VkCommandBuffer buffer;
};

struct Frame: public nb::intrusive_base {
    Frame(nb::ref<Window> window, gfx::Frame& frame);

    gfx::Frame& frame;
    nb::ref<CommandBuffer> command_buffer;
    nb::ref<Window> window;
    nb::ref<Image> image;
};

struct Window: public nb::intrusive_base {
    struct FrameManager {
        FrameManager(nb::ref<Window> window,
                     std::vector<std::tuple<nb::ref<Semaphore>, VkPipelineStageFlagBits>> additional_wait_semaphores,
                     std::vector<nb::ref<Semaphore>> additional_signal_semaphores)
            : window(window)
            , additional_wait_semaphores(std::move(additional_wait_semaphores))
            , additional_signal_semaphores(std::move(additional_signal_semaphores))
        {
        }

        nb::ref<Frame> enter() {
            frame = window->begin_frame();
            return frame;
        }

        void exit(nb::object, nb::object, nb::object) {
            window->end_frame(*frame, additional_wait_semaphores, additional_signal_semaphores);
        }

        nb::ref<Frame> frame;
        nb::ref<Window> window;
        std::vector<std::tuple<nb::ref<Semaphore>, VkPipelineStageFlagBits>> additional_wait_semaphores;
        std::vector<nb::ref<Semaphore>> additional_signal_semaphores;
    };

    FrameManager frame(std::vector<std::tuple<nb::ref<Semaphore>, VkPipelineStageFlagBits>> additional_wait_semaphores, std::vector<nb::ref<Semaphore>> additional_signal_semaphores) {
        return FrameManager(this, std::move(additional_wait_semaphores), std::move(additional_signal_semaphores));
    }

    Window(nb::ref<Context> ctx, const std::string& name, u32 width, u32 height)
        : ctx(ctx)
    {
        if (CreateWindowWithSwapchain(&window, ctx->vk, name.c_str(), width, height) != gfx::Result::SUCCESS) {
            throw std::runtime_error("Failed to create window");
        }
    }

    void set_callbacks(
        Function<void()> draw,
        Function<void(glm::ivec2)> mouse_move_event,
        Function<void(glm::ivec2, gfx::MouseButton, gfx::Action, gfx::Modifiers)> mouse_button_event,
        Function<void(glm::ivec2, glm::ivec2)> mouse_scroll_event,
        Function<void(gfx::Key, gfx::Action, gfx::Modifiers)> key_event
    )
    {
        this->draw               = std::move(draw);
        this->mouse_move_event   = mouse_move_event;
        this->mouse_button_event = mouse_button_event;
        this->mouse_scroll_event = mouse_scroll_event;
        this->key_event          = key_event;
        this->draw               = draw;

        gfx::SetWindowCallbacks(&window, {
                .mouse_move_event = [this](glm::ivec2 p) {
                    try {
                        if(this->mouse_move_event)
                            this->mouse_move_event(p);
                    } catch (nb::python_error &e) {
                        e.restore();
                    }
                },
                .mouse_button_event = [this] (glm::ivec2 p, gfx::MouseButton b, gfx::Action a, gfx::Modifiers m) {
                    try {
                        if(this->mouse_button_event)
                            this->mouse_button_event(p, b, a, m);
                    } catch (nb::python_error &e) {
                        e.restore();
                    }
                },
                .mouse_scroll_event = [this] (glm::ivec2 p, glm::ivec2 s) {
                    try {
                        if(this->mouse_scroll_event)
                            this->mouse_scroll_event(p, s);
                    } catch (nb::python_error &e) {
                        e.restore();
                    }
                },
                .key_event = [this] (gfx::Key k, gfx::Action a, gfx::Modifiers m) {
                    try {
                        if(this->key_event)
                            this->key_event(k, a, m);
                    } catch (nb::python_error &e) {
                        e.restore();
                    }
                },
                .draw = [this] () {
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
        gfx::Frame& frame = gfx::WaitForFrame(&window, ctx->vk);
        gfx::Result ok = gfx::AcquireImage(&frame, &window, ctx->vk);
        if (ok != gfx::Result::SUCCESS) {
            throw std::runtime_error("Failed to acquire next image");
        }
        return new Frame(this, frame);
    }

    void end_frame(Frame& frame, const std::vector<std::tuple<nb::ref<Semaphore>, VkPipelineStageFlagBits>>& additional_wait_semaphores, const std::vector<nb::ref<Semaphore>>& additional_signal_semaphores)
    {
        VkResult vkr;

        // TODO: make this throw if not called after begin in the same frame
        if(additional_wait_semaphores.empty() && additional_signal_semaphores.empty()) {
            vkr = gfx::Submit(frame.frame, ctx->vk, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        } else {
            Array<VkSemaphore> wait_semaphores(additional_wait_semaphores.size() + 1);
            Array<VkPipelineStageFlags> wait_stages(additional_wait_semaphores.size() + 1);
            for(usize i = 0; i < additional_wait_semaphores.size(); i++) {
                wait_semaphores[i] = std::get<0>(additional_wait_semaphores[i])->semaphore;
                wait_stages[i] = std::get<1>(additional_wait_semaphores[i]);
            }
            wait_semaphores[additional_wait_semaphores.size()] = frame.frame.acquire_semaphore;
            wait_stages[additional_wait_semaphores.size()] = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

            Array<VkSemaphore> signal_semaphores(additional_signal_semaphores.size() + 1);
            for(usize i = 0; i < additional_signal_semaphores.size(); i++) {
                signal_semaphores[i] = additional_signal_semaphores[i]->semaphore;
            }
            signal_semaphores[additional_signal_semaphores.size()] = frame.frame.release_semaphore;

            vkr = gfx::SubmitQueue(ctx->vk.queue, {
                .cmd = { frame.frame.command_buffer },
                .wait_semaphores = Span(wait_semaphores),
                .wait_stages = Span(wait_stages),
                .signal_semaphores = Span(signal_semaphores),
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

    ~Window()
    {
        gfx::WaitIdle(ctx->vk);
        gfx::DestroyWindowWithSwapchain(&window, ctx->vk);
    }

    bool should_close() {
        return gfx::ShouldClose(window);
    }

    nb::ref<Context> ctx;
    gfx::Window window;
    std::vector<Frame> frames;

    Function<void()> draw;
    Function<void(glm::ivec2)> mouse_move_event;
    Function<void(glm::ivec2, gfx::MouseButton, gfx::Action, gfx::Modifiers)> mouse_button_event;
    Function<void(glm::ivec2, glm::ivec2)> mouse_scroll_event;
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
        nb::handle key_event          = nb::find(w->mouse_scroll_event);

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

        // Clear the cycle!
        w->ctx.reset();
        w->draw                = nullptr;
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

Frame::Frame(nb::ref<Window> window, gfx::Frame& frame)
    : window(window)
    , frame(frame)
{
    image = new Image(window->ctx, frame.current_image, frame.current_image_view);
    command_buffer = new CommandBuffer(window->ctx, frame.command_pool, frame.command_buffer, false);
}

struct Gui: public nb::intrusive_base {
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

    ~Gui()
    {
        gfx::WaitIdle(window->ctx->vk);
        gui::DestroyImGuiImpl(&imgui_impl, window->ctx->vk);
    }

    nb::ref<Window> window;
    gui::ImGuiImpl imgui_impl;

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

struct DescriptorSet: public nb::intrusive_base {
    DescriptorSet(nb::ref<Context> ctx, const std::vector<DescriptorSetEntry>& entries, VkDescriptorBindingFlagBits flags)
        : ctx(ctx)
    {
        VkResult vkr = gfx::CreateDescriptorSet(&set, ctx->vk, {
            .entries = ArrayView((gfx::DescriptorSetEntryDesc*)entries.data(), entries.size()),
            .flags = (VkDescriptorBindingFlags)flags,
        });
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor set");
        }
    }

    void write_buffer(const Buffer& buffer, VkDescriptorType type, u32 binding, u32 element) {
        gfx::WriteBufferDescriptor(set.set, ctx->vk, {
            .buffer = buffer.buffer.buffer,
            .type = type,
            .binding = binding,
            .element = element,
        });
    };

    ~DescriptorSet()
    {
        destroy();
    }

    void destroy()
    {
        gfx::DestroyDescriptorSet(&set, ctx->vk);
    }

    nb::ref<Context> ctx;
    gfx::DescriptorSet set;
};

struct Shader: public nb::intrusive_base {
    Shader(nb::ref<Context> ctx, const nb::bytes& code)
        : ctx(ctx)
    {
        VkResult vkr = gfx::CreateShader(&shader, ctx->vk, ArrayView<u8>((u8*)code.data(), code.size()));
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to create shader");
        }
    }

    ~Shader() {
        destroy();
    }

    void destroy() {
        gfx::DestroyShader(&shader, ctx->vk);
    }

    gfx::Shader shader;
    nb::ref<Context> ctx;
};

struct PipelineStage: public nb::intrusive_base {
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
    enum class InputRate {
        Vertex,
        Instance,
    };

    VertexBinding(u32 binding, u32 stride, InputRate input_rate)
        : gfx::VertexBindingDesc {
            .binding = binding,
            .stride = stride,
            .input_rate = (VkVertexInputRate)input_rate,
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

struct InputAssembly: gfx::InputAssemblyDesc{
    InputAssembly(VkPrimitiveTopology primitive_topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, bool primitive_restart_enable = false)
        : gfx::InputAssemblyDesc {
              .primitive_topology = (VkPrimitiveTopology)primitive_topology,
              .primitive_restart_enable = primitive_restart_enable,
          }
    {
    }
};
static_assert(sizeof(InputAssembly) == sizeof(gfx::InputAssemblyDesc));

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

struct GraphicsPipeline: nb::intrusive_base {
    GraphicsPipeline(nb::ref<Context> ctx,
        const std::vector<nb::ref<PipelineStage>>& stages,
        const std::vector<VertexBinding>& vertex_bindings,
        const std::vector<VertexAttribute>& vertex_attributes,
        InputAssembly input_assembly,
        const std::vector<PushConstantsRange>& push_constant_ranges,
        const std::vector<nb::ref<DescriptorSet>>& descriptor_sets,
        u32 samples,
        const std::vector<Attachment>& attachments,
        Depth depth
        )
        : ctx(ctx)
    {
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
            .samples = (VkSampleCountFlagBits)samples,
            .depth = depth,
            .push_constants = ArrayView((gfx::PushConstantsRangeDesc*)push_constant_ranges.data(), push_constant_ranges.size()),
            .descriptor_sets = ArrayView(d),
            .attachments = ArrayView((gfx::AttachmentDesc*)attachments.data(), attachments.size()),
        });

        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to create graphics pipeline");
        }
    }

    ~GraphicsPipeline() {
        destroy();
    }

    void destroy() {
        gfx::DestroyGraphicsPipeline(&pipeline, ctx->vk);
    }

    gfx::GraphicsPipeline pipeline;
    nb::ref<Context> ctx;
};

void CommandBuffer::bind_pipeline_state(
    const GraphicsPipeline& pipeline,
    const std::vector<nb::ref<DescriptorSet>> descriptor_sets,
    std::optional<nb::bytes> push_constants,
    const std::vector<nb::ref<Buffer>> vertex_buffers,
    std::optional<nb::ref<Buffer>> index_buffer,
    std::array<u32, 4> viewport,
    std::array<u32, 4> scissors
)
{
    // Pipeline
    vkCmdBindPipeline(buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline.pipeline);

    // Descriptor sets
    if(descriptor_sets.size() > 0) {
        Array<VkDescriptorSet> sets(descriptor_sets.size());
        for(usize i = 0; i < sets.length; i++) {
            sets[i] = descriptor_sets[i]->set.set;
        }
        vkCmdBindDescriptorSets(buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline.layout, 0, sets.length, sets.data, 0, 0);
    }

    // Push constants
    if(push_constants.has_value()) {
        vkCmdPushConstants(buffer, pipeline.pipeline.layout, VK_SHADER_STAGE_ALL, 0, push_constants->size(), push_constants->data());
    }

    // Vertex buffers
    if(vertex_buffers.size() > 0) {
        Array<VkDeviceSize> offsets(vertex_buffers.size());
        Array<VkBuffer> buffers(vertex_buffers.size());
        for(usize i = 0; i < vertex_buffers.size(); i++) {
            offsets[i] = 0;
            buffers[i] = vertex_buffers[i]->buffer.buffer;
        }
        vkCmdBindVertexBuffers(buffer, 0, buffers.length, buffers.data, offsets.data);
    }


    // Index buffers
    if(index_buffer.has_value()) {
        vkCmdBindIndexBuffer(buffer, index_buffer.value()->buffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    }

    // Viewport
    VkViewport vp = {};
    vp.x = (float)viewport[0];
    vp.y = (float)viewport[1];
    vp.width = (float)viewport[2];
    vp.height = (float)viewport[3];
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(buffer, 0, 1, &vp);

    // Scissors
    VkRect2D scissor = {};
    scissor.offset.x = scissors[0];
    scissor.offset.y = scissors[1];
    scissor.extent.width = scissors[2];
    scissor.extent.height = scissors[3];
    vkCmdSetScissor(buffer, 0, 1, &scissor);
}

void gfx_create_bindings(nb::module_& m)
{
    nb::class_<Context>(m, "Context",
        nb::intrusive_ptr<Context>([](Context *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<>())
    ;

    nb::class_<Frame>(m, "Frame",
        nb::intrusive_ptr<Frame>([](Frame *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def_ro("command_buffer", &Frame::command_buffer)
        .def_ro("image", &Frame::image)
    ;

    nb::class_<Window>(m, "Window",
        nb::type_slots(window_tp_slots),
        nb::intrusive_ptr<Window>([](Window *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<nb::ref<Context>, const::std::string&, u32, u32>(), nb::arg("ctx"), nb::arg("title"), nb::arg("width"), nb::arg("height"))
        .def("should_close", &Window::should_close)
        .def("set_callbacks", &Window::set_callbacks,
            nb::arg("draw"),
            nb::arg("mouse_move_event").none() = nb::none(),
            nb::arg("mouse_button_event").none() = nb::none(),
            nb::arg("mouse_scroll_event").none() = nb::none(),
            nb::arg("key_event").none() = nb::none())
        .def("reset_callbacks", &Window::reset_callbacks)
        .def("update_swapchain", &Window::update_swapchain)
        .def("begin_frame", &Window::begin_frame)
        .def("end_frame", &Window::end_frame,
            nb::arg("frame"),
            nb::arg("additional_wait_semaphores") = nb::list(),
            nb::arg("additional_signal_semaphores") = nb::list()
        )
        .def("frame", &Window::frame,
            nb::arg("additional_wait_semaphores") = nb::list(),
            nb::arg("additional_signal_semaphores") = nb::list()
        )
        .def_prop_ro("swapchain_format", [](Window& w) -> VkFormat { return w.window.swapchain_format; })
        .def_prop_ro("fb_width", [](Window& w) -> u32 { return w.window.fb_width; })
        .def_prop_ro("fb_height", [](Window& w) -> u32 { return w.window.fb_height; })
        .def_prop_ro("num_frames", [](Window& w) -> usize { return w.window.frames.length; })
    ;

    nb::class_<Window::FrameManager>(m, "WindowFrame")
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
    ;

    nb::enum_<gfx::Action>(m, "Action")
        .value("NONE", gfx::Action::None)
        .value("RELEASE", gfx::Action::Release)
        .value("PRESS", gfx::Action::Press)
        .value("REPEAT", gfx::Action::Repeat)
    ;

    nb::enum_<gfx::Key>(m, "Key")
        .value("ESCAPE", gfx::Key::Escape)
        .value("SPACE", gfx::Key::Space)
        .value("PERIOD", gfx::Key::Period)
        .value("COMMA", gfx::Key::Comma)
    ;

    nb::enum_<gfx::MouseButton>(m, "MouseButton")
        .value("NONE",   gfx::MouseButton::None)
        .value("LEFT",   gfx::MouseButton::Left)
        .value("RIGHT",  gfx::MouseButton::Right)
        .value("MIDDLE", gfx::MouseButton::Middle)
    ;

    nb::enum_<gfx::Modifiers>(m, "Modifiers", nb::is_flag(), nb::is_arithmetic())
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
    ;

    nb::enum_<VkImageUsageFlagBits>(m, "ImageUsageFlags", nb::is_arithmetic() , nb::is_flag())
        .value("TRANSFER_SRC",                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
        .value("TRANSFER_DST",                         VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .value("SAMPLED",                              VK_IMAGE_USAGE_SAMPLED_BIT)
        .value("STORAGE",                              VK_IMAGE_USAGE_STORAGE_BIT)
        .value("COLOR_ATTACHMENT",                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
        .value("DEPTH_STENCIL_ATTACHMENT",             VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
        // .value("TRANSIENT_ATTACHMENT",                 VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT)
        // .value("INPUT_ATTACHMENT",                     VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT)
        // .value("VIDEO_DECODE_DST_KHR",                 VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR)
        // .value("VIDEO_DECODE_SRC_KHR",                 VK_IMAGE_USAGE_VIDEO_DECODE_SRC_BIT_KHR)
        // .value("VIDEO_DECODE_DPB_KHR",                 VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR)
        // .value("FRAGMENT_DENSITY_MAP_EXT",             VK_IMAGE_USAGE_FRAGMENT_DENSITY_MAP_BIT_EXT)
        // .value("FRAGMENT_SHADING_RATE_ATTACHMENT_KHR", VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR)
        // .value("HOST_TRANSFER_EXT",                    VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT)
        // .value("VIDEO_ENCODE_DST_KHR",                 VK_IMAGE_USAGE_VIDEO_ENCODE_DST_BIT_KHR)
        // .value("VIDEO_ENCODE_SRC_KHR",                 VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR)
        // .value("VIDEO_ENCODE_DPB_KHR",                 VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR)
        // .value("ATTACHMENT_FEEDBACK_LOOP_EXT",         VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT)
        // .value("INVOCATION_MASK_HUAWEI",               VK_IMAGE_USAGE_INVOCATION_MASK_BIT_HUAWEI)
        // .value("SAMPLE_WEIGHT_QCOM",                   VK_IMAGE_USAGE_SAMPLE_WEIGHT_BIT_QCOM)
        // .value("SAMPLE_BLOCK_MATCH_QCOM",              VK_IMAGE_USAGE_SAMPLE_BLOCK_MATCH_BIT_QCOM)
        // .value("SHADING_RATE_IMAGE_NV",                VK_IMAGE_USAGE_SHADING_RATE_IMAGE_BIT_NV)
    ;

    nb::class_<GfxObject>(m, "GfxObject",
        nb::intrusive_ptr<GfxObject>([](GfxObject *o, PyObject *po) noexcept { o->set_self_py(po); }))
    ;

    nb::class_<Buffer, GfxObject>(m, "Buffer")
        .def(nb::init<nb::ref<Context>, size_t, VkBufferUsageFlags, gfx::AllocPresets::Type>(), nb::arg("ctx"), nb::arg("size"), nb::arg("usage_flags"), nb::arg("alloc_type"))
        .def("destroy", &Buffer::destroy)
        .def_static("from_data", &Buffer::from_data)
        .def_prop_ro("view", [] (Buffer& buffer) {
            return nb::ndarray<u8, nb::numpy, nb::shape<-1>>(buffer.buffer.map.data, {buffer.buffer.map.length}, buffer.self_py());
        }, nb::rv_policy::reference_internal)
    ;

    nb::class_<ExternalBuffer, Buffer>(m, "ExternalBuffer")
        .def(nb::init<nb::ref<Context>, size_t, VkBufferUsageFlags, gfx::AllocPresets::Type>(), nb::arg("ctx"), nb::arg("size"), nb::arg("usage_flags"), nb::arg("alloc_type"))
        .def("destroy", &ExternalBuffer::destroy)
        .def_prop_ro("handle", [] (ExternalBuffer& buffer) { return (u64)buffer.handle; });
    ;

    nb::class_<Image, GfxObject>(m, "Image")
        .def(nb::init<nb::ref<Context>, u32, u32, VkFormat, VkImageUsageFlags, gfx::AllocPresets::Type, int>(), nb::arg("ctx"), nb::arg("width"), nb::arg("height"), nb::arg("format"), nb::arg("usage_flags"), nb::arg("alloc_type"), nb::arg("samples") = 1)
        .def("destroy", &Image::destroy)
        // .def_static("from_data", &Image::from_data)
    ;

    nb::class_<Semaphore, GfxObject>(m, "Semaphore")
        .def(nb::init<nb::ref<Context>>(), nb::arg("ctx"))
        .def("destroy", &Semaphore::destroy)
    ;

    nb::class_<ExternalSemaphore, Semaphore>(m, "ExternalSemaphore")
        .def(nb::init<nb::ref<Context>>(), nb::arg("ctx"))
        .def("destroy", &ExternalSemaphore::destroy)
        .def_prop_ro("handle", [] (ExternalSemaphore& semaphore) { return (u64)semaphore.handle; });
    ;

    nb::enum_<ImageUsage>(m, "ImageUsage")
        .value("NONE", ImageUsage::None)
        .value("IMAGE", ImageUsage::Image)
        .value("IMAGE_READ_ONLY", ImageUsage::ImageReadOnly)
        .value("IMAGE_WRITE_ONLY", ImageUsage::ImageWriteOnly)
        .value("COLOR_ATTACHMENT", ImageUsage::ColorAttachment)
        .value("COLOR_ATTACHMENT_WRITE_ONLY", ImageUsage::ColorAttachmentWriteOnly)
        .value("DEPTH_STENCIL_ATTACHMENT", ImageUsage::DepthStencilAttachment)
        .value("DEPTH_STENCIL_ATTACHMENT_READ_ONLY", ImageUsage::DepthStencilAttachmentReadOnly)
        .value("DEPTH_STENCIL_ATTACHMENT_WRITE_ONLY", ImageUsage::DepthStencilAttachmentWriteOnly)
        .value("PRESENT", ImageUsage::Present)
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

    nb::class_<RenderingAttachment>(m, "RenderingAttachment")
        .def(nb::init<nb::ref<Image>, VkAttachmentLoadOp, VkAttachmentStoreOp, std::array<float, 4>, std::optional<nb::ref<Image>>, VkResolveModeFlagBits>(),
            nb::arg("image"), nb::arg("load_op"), nb::arg("store_op"), nb::arg("clear") = std::array<float,4>({0.0f, 0.0f, 0.0f, 0.0f}), nb::arg("resolve_image").none() = nb::none(), nb::arg("resolve_mode") = VK_RESOLVE_MODE_NONE)
    ;

    nb::class_<DepthAttachment>(m, "DepthAttachment")
        .def(nb::init<nb::ref<Image>, VkAttachmentLoadOp, VkAttachmentStoreOp, float>(),
            nb::arg("image"), nb::arg("load_op"), nb::arg("store_op"), nb::arg("clear") = 0.0f)
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

    nb::class_<CommandBuffer::RenderingManager>(m, "RenderingManager")
        .def("__enter__", &CommandBuffer::RenderingManager::enter)
        .def("__exit__", &CommandBuffer::RenderingManager::exit, nb::arg("exc_type").none(), nb::arg("exc_val").none(), nb::arg("exc_tb").none())
    ;

    nb::class_<CommandBuffer, GfxObject>(m, "CommandBuffer")
        .def("__enter__", &CommandBuffer::enter)
        .def("__exit__", &CommandBuffer::exit, nb::arg("exc_type").none(), nb::arg("exc_val").none(), nb::arg("exc_tb").none())
        .def("begin", &CommandBuffer::begin)
        .def("end", &CommandBuffer::end)
        .def("use_image", &CommandBuffer::use_image, nb::arg("image"), nb::arg("usage"))
        .def("begin_rendering", &CommandBuffer::begin_rendering, nb::arg("viewport"), nb::arg("color_attachments"), nb::arg("depth").none() = nb::none())
        .def("end_rendering", &CommandBuffer::end_rendering)
        .def("rendering", &CommandBuffer::rendering, nb::arg("viewport"), nb::arg("color_attachments"), nb::arg("depth").none() = nb::none())
        .def("bind_pipeline_state", &CommandBuffer::bind_pipeline_state,
            nb::arg("pipeline"),
            nb::arg("descriptor_sets") = std::vector<nb::ref<DescriptorSet>>(),
            nb::arg("push_constants") = std::optional<nb::bytes>(),
            nb::arg("vertex_buffers") = std::vector<nb::ref<Buffer>>(),
            nb::arg("index_buffer") = std::optional<nb::ref<Buffer>>(),
            nb::arg("viewport") = std::array<float, 4>(),
            nb::arg("scissors") = std::array<float, 4>()
        )
        .def("draw_indexed", &CommandBuffer::draw_indexed,
            nb::arg("num_indices"),
            nb::arg("num_instances") = 1,
            nb::arg("first_index") = 0,
            nb::arg("vertex_offset") = 0,
            nb::arg("first_instanc") = 0
        )
    ;

    nb::class_<Shader>(m, "Shader",
        nb::intrusive_ptr<Shader>([](Shader *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<nb::ref<Context>, const nb::bytes&>(), nb::arg("ctx"), nb::arg("code"))
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

    nb::enum_<VertexBinding::InputRate>(m, "VertexInputRate")
        .value("VERTEX", VertexBinding::InputRate::Vertex)
        .value("INSTANCE", VertexBinding::InputRate::Instance)
    ;

    nb::class_<VertexBinding>(m, "VertexBinding")
        .def(nb::init<u32, u32, VertexBinding::InputRate>(), nb::arg("binding"), nb::arg("stride"), nb::arg("input_rate") = VertexBinding::InputRate::Vertex)
    ;

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

    nb::class_<DescriptorSet>(m, "DescriptorSet",
        nb::intrusive_ptr<DescriptorSet>([](DescriptorSet *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<nb::ref<Context>, const std::vector<DescriptorSetEntry>&, VkDescriptorBindingFlagBits>(), nb::arg("ctx"), nb::arg("entries"), nb::arg("flags") = VkDescriptorBindingFlagBits())
        .def("write_buffer", &DescriptorSet::write_buffer, nb::arg("buffer"), nb::arg("type"), nb::arg("binding"), nb::arg("element"))
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

    nb::class_<Depth>(m, "Depth")
        .def(nb::init<VkFormat, bool, bool, VkCompareOp>(),
            nb::arg("format") = VK_FORMAT_UNDEFINED,
            nb::arg("test") = false,
            nb::arg("write") = false,
            nb::arg("op") = VK_COMPARE_OP_LESS
        )
    ;

    nb::class_<GraphicsPipeline>(m, "GraphicsPipeline",
        nb::intrusive_ptr<GraphicsPipeline>([](GraphicsPipeline *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<nb::ref<Context>,
                const std::vector<nb::ref<PipelineStage>>&,
                const std::vector<VertexBinding>&,
                const std::vector<VertexAttribute>&,
                InputAssembly,
                const std::vector<PushConstantsRange>&,
                const std::vector<nb::ref<DescriptorSet>>&,
                u32,
                const std::vector<Attachment>&,
                Depth
            >(),
            nb::arg("ctx"),
            nb::arg("stages") = std::vector<nb::ref<PipelineStage>>(),
            nb::arg("vertex_bindings") = std::vector<VertexBinding>(),
            nb::arg("vertex_attributes") = std::vector<VertexAttribute>(),
            nb::arg("input_assembly") = InputAssembly(),
            nb::arg("push_constants_ranges") = std::vector<PushConstantsRange>(),
            nb::arg("descriptor_sets") = std::vector<nb::ref<DescriptorSet>>(),
            nb::arg("samples") = 1,
            nb::arg("attachments") = std::vector<Attachment>(),
            nb::arg("depth") = Depth(VK_FORMAT_UNDEFINED)
        )
        .def("destroy", &GraphicsPipeline::destroy)
    ;

    nb::enum_<gfx::SwapchainStatus>(m, "SwapchainStatus")
        .value("READY", gfx::SwapchainStatus::READY)
        .value("RESIZED", gfx::SwapchainStatus::RESIZED)
        .value("MINIMIZED", gfx::SwapchainStatus::MINIMIZED)
    ;

    m.def("process_events", &gfx::ProcessEvents, nb::arg("wait"));
    m.def("wait_idle", [](Context& ctx) { gfx::WaitIdle(ctx.vk); });
}