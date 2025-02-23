#include <nanobind/nanobind.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>
#include <nanobind/intrusive/counter.h>
#include <nanobind/intrusive/ref.h>
#include <nanobind/intrusive/counter.inl>

#if 0
#include <slang.h>
#include <slang-com-ptr.h>
#endif

#include <xpg/gfx.h>
#include <xpg/gui.h>

#include "function.h"

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

        result = gfx::CreateContext(&vk, {
            .minimum_api_version = (u32)VK_API_VERSION_1_3,
            .instance_extensions = instance_extensions,
            .device_extensions = device_extensions,
            .device_features = gfx::DeviceFeatures::DYNAMIC_RENDERING | gfx::DeviceFeatures::DESCRIPTOR_INDEXING | gfx::DeviceFeatures::SYNCHRONIZATION_2,
            .enable_validation_layer = true,
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

struct Window;

struct Frame {
    Frame(gfx::Frame& frame)
        : frame(frame)
    {
    }

    gfx::Frame& frame;
};

struct Window: public nb::intrusive_base {
    Window(nb::ref<Context> ctx, const std::string& name, u32 width, u32 height)
        : ctx(ctx)
    {
        if (CreateWindowWithSwapchain(&window, ctx->vk, name.c_str(), width, height) != gfx::Result::SUCCESS) {
            throw std::runtime_error("Failed to create window");
        }
    }

    void set_callbacks(Function<void()> draw)
    {
        gfx::SetWindowCallbacks(&window, {
                .draw = std::move(draw)
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

    Frame begin_frame()
    {
        // TODO: make this throw if called multiple times in a row befor end
        gfx::Frame& frame = gfx::WaitForFrame(&window, ctx->vk);
        gfx::Result ok = gfx::AcquireImage(&frame, &window, ctx->vk);
        if (ok != gfx::Result::SUCCESS) {
            throw std::runtime_error("Failed to acquire next image");
        }
        return Frame(frame);
    }

    void end_frame(Frame& frame)
    {
        // TODO: make this throw if not called after begin in the same frame

        VkResult vkr;
        vkr = gfx::Submit(frame.frame, ctx->vk, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
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

    // Garbage collection:

    static int tp_traverse(PyObject *self, visitproc visit, void *arg) {
        // Retrieve a pointer to the C++ instance associated with 'self' (never fails)
        Window *w = nb::inst_ptr<Window>(self);

        // If w->value has an associated CPython object, return it.
        // If not, value.ptr() will equal NULL, which is also fine.
        nb::handle ctx                = nb::find(w->ctx.get());
        nb::handle draw               = nb::find(w->window.callbacks.draw);
        // nb::handle mouse_move_event   = nb::find(w->window.callbacks.mouse_move_event);
        // nb::handle mouse_button_event = nb::find(w->window.callbacks.mouse_button_event);
        // nb::handle mouse_scroll_event = nb::find(w->window.callbacks.mouse_scroll_event);

        // Inform the Python GC about the instance (if non-NULL)
        Py_VISIT(ctx.ptr());
        Py_VISIT(draw.ptr());
        // Py_VISIT(mouse_move_event.ptr());
        // Py_VISIT(mouse_button_event.ptr());
        // Py_VISIT(mouse_scroll_event.ptr());

        return 0;
    }

    static int tp_clear(PyObject *self) {
        // Retrieve a pointer to the C++ instance associated with 'self' (never fails)
        Window *w = nb::inst_ptr<Window>(self);

        // Clear the cycle!
        w->ctx.reset();
        w->window.callbacks.draw = nullptr;
        // w->window.callbacks.mouse_move_event = nullptr;
        // w->window.callbacks.mouse_button_event = nullptr;
        // w->window.callbacks.mouse_scroll_event = nullptr;

        return 0;
    }

    // Slot data structure referencing the above two functions
    static constexpr PyType_Slot tp_slots[] = {
        { Py_tp_traverse, (void *) Window::tp_traverse },
        { Py_tp_clear, (void *) Window::tp_clear },
        { 0, nullptr }
    };
};


struct Gui: public nb::intrusive_base {
    Gui(nb::ref<Window> window)
        : window(window)
        , ctx(window->ctx)
    {
        gui::CreateImGuiImpl(&imgui_impl, window->window, ctx->vk, {});
    }

    void begin_frame()
    {
        gui::BeginFrame();
    }

    void end_frame()
    {
        gui::EndFrame();
    }

    void render(Frame& frame_obj) //TODO: Ideally this would be just a command buffer, but need to decide how to expose lower level gfx
    {
        gfx::Context& vk = ctx->vk;
        gfx::Frame& frame = frame_obj.frame;
        gfx::Window& window = this->window->window;

        gfx::CmdImageBarrier(frame.command_buffer, {
            .image = frame.current_image,
            .src_stage = VK_PIPELINE_STAGE_2_NONE,
            .dst_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .src_access = 0,
            .dst_access = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
            .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            });

        VkClearColorValue color = { 0.1f, 0.1f, 0.1f, 1.0f };
        VkRenderingAttachmentInfo attachment_info = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        attachment_info.imageView = frame.current_image_view;
        attachment_info.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        attachment_info.resolveMode = VK_RESOLVE_MODE_NONE;
        attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachment_info.clearValue.color = color;

        VkRenderingInfo rendering_info = { VK_STRUCTURE_TYPE_RENDERING_INFO };
        rendering_info.renderArea.extent.width = window.fb_width;
        rendering_info.renderArea.extent.height = window.fb_height;
        rendering_info.layerCount = 1;
        rendering_info.viewMask = 0;
        rendering_info.colorAttachmentCount = 1;
        rendering_info.pColorAttachments = &attachment_info;
        rendering_info.pDepthAttachment = 0;
        vkCmdBeginRenderingKHR(frame.command_buffer, &rendering_info);

        // Draw GUI
        gui::Render(frame.command_buffer);

        vkCmdEndRenderingKHR(frame.command_buffer);

        gfx::CmdImageBarrier(frame.command_buffer, {
            .image = frame.current_image,
            .src_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dst_stage = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
            .src_access = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .dst_access = 0,
            .old_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .new_layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            });
    }

    ~Gui()
    {
        gfx::WaitIdle(ctx->vk);
        gui::DestroyImGuiImpl(&imgui_impl, ctx->vk);
    }

    nb::ref<Window> window;
    nb::ref<Context> ctx;
    gui::ImGuiImpl imgui_impl;

    // Garbage collection:

    static int tp_traverse(PyObject *self, visitproc visit, void *arg) {
        // Retrieve a pointer to the C++ instance associated with 'self' (never fails)
        Gui *g = nb::inst_ptr<Gui>(self);

        // If w->value has an associated CPython object, return it.
        // If not, value.ptr() will equal NULL, which is also fine.
        nb::handle window = nb::find(g->window.get());
        nb::handle ctx = nb::find(g->ctx.get());

        // Inform the Python GC about the instance (if non-NULL)
        Py_VISIT(window.ptr());
        Py_VISIT(ctx.ptr());

        return 0;
    }

    static int tp_clear(PyObject *self) {
        // Retrieve a pointer to the C++ instance associated with 'self' (never fails)
        Gui *g = nb::inst_ptr<Gui>(self);

        // Clear the cycle!
        g->window.reset();
        g->ctx.reset();

        return 0;
    }

    // Slot data structure referencing the above two functions
    static constexpr PyType_Slot tp_slots[] = {
        { Py_tp_traverse, (void *) Gui::tp_traverse },
        { Py_tp_clear, (void *) Gui::tp_clear },
        { 0, nullptr }
    };
};

struct Buffer: public nb::intrusive_base {
    enum class UsageFlags {
        TransferSrc = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        TransferDst = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        Uniform = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        Storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        Index =  VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        Vertex = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        Indirect = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        AccelerationStructureInput = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        AccelerationStructureStorage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
    };

    Buffer(nb::ref<Context> ctx, usize size, UsageFlags usage_flags, gfx::AllocPresets::Type alloc_type)
        : ctx(ctx)
    {
        VkResult vkr = gfx::CreateBuffer(&buffer, ctx->vk, size, {
            .usage = (VkBufferUsageFlags)usage_flags,
            .alloc = gfx::AllocPresets::Types[(size_t)alloc_type],
        });
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to create buffer");
        }
    }

    Buffer(nb::ref<Context> ctx, nb::bytes data, UsageFlags usage_flags, gfx::AllocPresets::Type alloc_type)
        : ctx(ctx)
    {
        VkResult vkr = gfx::CreateBufferFromData(&buffer, ctx->vk, ArrayView<u8>((u8*)data.data(), data.size()), {
            .usage = (VkBufferUsageFlags)usage_flags,
            .alloc = gfx::AllocPresets::Types[(size_t)alloc_type],
        });
        if (vkr != VK_SUCCESS) {
            throw std::runtime_error("Failed to create buffer");
        }
    }

    ~Buffer() {
        destroy();
    }

    void destroy() {
        gfx::DestroyBuffer(&buffer, ctx->vk);
    }

    gfx::Buffer buffer;
    nb::ref<Context> ctx;
};

enum DescriptorType {
    Sampler = VK_DESCRIPTOR_TYPE_SAMPLER,
    CombinedImageSampler = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    SampledImage = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
    StorageImage = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
    UniformTexelBuffer = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
    StorageTexelBuffer = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
    UniformBuffer = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    StorageBuffer = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    UniformBufferDynamic = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
    StorageBufferDynamic = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC,
    InputAttachment = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
    InlineUniformBlock = VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK,
    AccelerationStructure = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
    SampleWeightImage = VK_DESCRIPTOR_TYPE_SAMPLE_WEIGHT_IMAGE_QCOM,
    BlockMatchImage = VK_DESCRIPTOR_TYPE_BLOCK_MATCH_IMAGE_QCOM,
    Mutable = VK_DESCRIPTOR_TYPE_MUTABLE_EXT,
};

struct DescriptorSetEntry: gfx::DescriptorSetEntryDesc {
    DescriptorSetEntry(u32 count, DescriptorType type)
        : gfx::DescriptorSetEntryDesc {
            .count = count,
            .type = (VkDescriptorType)type
        }
    {
    };
};
static_assert(sizeof(DescriptorSetEntry) == sizeof(gfx::DescriptorSetEntryDesc));

enum class DescriptorBindingFlags {
    UpdateAfterBind          = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
    UpdateUnusedWhilePending = VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT,
    PartiallyBound           = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
    VariableDescriptorCount  = VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT,
};

struct DescriptorSet: public nb::intrusive_base {
    DescriptorSet(nb::ref<Context> ctx, const std::vector<DescriptorSetEntry>& entries, DescriptorBindingFlags flags)
        : ctx(ctx)
    {
        gfx::CreateDescriptorSet(&set, ctx->vk, {
            .entries = ArrayView((gfx::DescriptorSetEntryDesc*)entries.data(), entries.size()),
        });
    }

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
    Shader(nb::ref<Context> ctx, nb::bytes code)
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
    enum class Stage {
        Vertex = VK_SHADER_STAGE_VERTEX_BIT,
        TessellationControl = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
        TessellationEvaluation = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
        Geometry = VK_SHADER_STAGE_GEOMETRY_BIT,
        Fragment = VK_SHADER_STAGE_FRAGMENT_BIT,
        Compute = VK_SHADER_STAGE_COMPUTE_BIT,
        Raygen = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
        AnyHit = VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
        ClosestHit = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
        Miss = VK_SHADER_STAGE_MISS_BIT_KHR,
        Intersection = VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
        Callable = VK_SHADER_STAGE_CALLABLE_BIT_KHR,
        Task = VK_SHADER_STAGE_TASK_BIT_EXT,
        Mesh = VK_SHADER_STAGE_MESH_BIT_EXT,
    };

    PipelineStage(nb::ref<Shader> shader, Stage stage, std::string entry)
        : shader(shader)
        , stage(stage)
        , entry(std::move(entry)) {
    };

    nb::ref<Shader> shader;
    Stage stage;
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

enum class Format {
    UNDEFINED =                                      VK_FORMAT_UNDEFINED,
    R4G4_UNORM_PACK8 =                               VK_FORMAT_R4G4_UNORM_PACK8,
    R4G4B4A4_UNORM_PACK16 =                          VK_FORMAT_R4G4B4A4_UNORM_PACK16,
    B4G4R4A4_UNORM_PACK16 =                          VK_FORMAT_B4G4R4A4_UNORM_PACK16,
    R5G6B5_UNORM_PACK16 =                            VK_FORMAT_R5G6B5_UNORM_PACK16,
    B5G6R5_UNORM_PACK16 =                            VK_FORMAT_B5G6R5_UNORM_PACK16,
    R5G5B5A1_UNORM_PACK16 =                          VK_FORMAT_R5G5B5A1_UNORM_PACK16,
    B5G5R5A1_UNORM_PACK16 =                          VK_FORMAT_B5G5R5A1_UNORM_PACK16,
    A1R5G5B5_UNORM_PACK16 =                          VK_FORMAT_A1R5G5B5_UNORM_PACK16,
    R8_UNORM =                                       VK_FORMAT_R8_UNORM,
    R8_SNORM =                                       VK_FORMAT_R8_SNORM,
    R8_USCALED =                                     VK_FORMAT_R8_USCALED,
    R8_SSCALED =                                     VK_FORMAT_R8_SSCALED,
    R8_UINT =                                        VK_FORMAT_R8_UINT,
    R8_SINT =                                        VK_FORMAT_R8_SINT,
    R8_SRGB =                                        VK_FORMAT_R8_SRGB,
    R8G8_UNORM =                                     VK_FORMAT_R8G8_UNORM,
    R8G8_SNORM =                                     VK_FORMAT_R8G8_SNORM,
    R8G8_USCALED =                                   VK_FORMAT_R8G8_USCALED,
    R8G8_SSCALED =                                   VK_FORMAT_R8G8_SSCALED,
    R8G8_UINT =                                      VK_FORMAT_R8G8_UINT,
    R8G8_SINT =                                      VK_FORMAT_R8G8_SINT,
    R8G8_SRGB =                                      VK_FORMAT_R8G8_SRGB,
    R8G8B8_UNORM =                                   VK_FORMAT_R8G8B8_UNORM,
    R8G8B8_SNORM =                                   VK_FORMAT_R8G8B8_SNORM,
    R8G8B8_USCALED =                                 VK_FORMAT_R8G8B8_USCALED,
    R8G8B8_SSCALED =                                 VK_FORMAT_R8G8B8_SSCALED,
    R8G8B8_UINT =                                    VK_FORMAT_R8G8B8_UINT,
    R8G8B8_SINT =                                    VK_FORMAT_R8G8B8_SINT,
    R8G8B8_SRGB =                                    VK_FORMAT_R8G8B8_SRGB,
    B8G8R8_UNORM =                                   VK_FORMAT_B8G8R8_UNORM,
    B8G8R8_SNORM =                                   VK_FORMAT_B8G8R8_SNORM,
    B8G8R8_USCALED =                                 VK_FORMAT_B8G8R8_USCALED,
    B8G8R8_SSCALED =                                 VK_FORMAT_B8G8R8_SSCALED,
    B8G8R8_UINT =                                    VK_FORMAT_B8G8R8_UINT,
    B8G8R8_SINT =                                    VK_FORMAT_B8G8R8_SINT,
    B8G8R8_SRGB =                                    VK_FORMAT_B8G8R8_SRGB,
    R8G8B8A8_UNORM =                                 VK_FORMAT_R8G8B8A8_UNORM,
    R8G8B8A8_SNORM =                                 VK_FORMAT_R8G8B8A8_SNORM,
    R8G8B8A8_USCALED =                               VK_FORMAT_R8G8B8A8_USCALED,
    R8G8B8A8_SSCALED =                               VK_FORMAT_R8G8B8A8_SSCALED,
    R8G8B8A8_UINT =                                  VK_FORMAT_R8G8B8A8_UINT,
    R8G8B8A8_SINT =                                  VK_FORMAT_R8G8B8A8_SINT,
    R8G8B8A8_SRGB =                                  VK_FORMAT_R8G8B8A8_SRGB,
    B8G8R8A8_UNORM =                                 VK_FORMAT_B8G8R8A8_UNORM,
    B8G8R8A8_SNORM =                                 VK_FORMAT_B8G8R8A8_SNORM,
    B8G8R8A8_USCALED =                               VK_FORMAT_B8G8R8A8_USCALED,
    B8G8R8A8_SSCALED =                               VK_FORMAT_B8G8R8A8_SSCALED,
    B8G8R8A8_UINT =                                  VK_FORMAT_B8G8R8A8_UINT,
    B8G8R8A8_SINT =                                  VK_FORMAT_B8G8R8A8_SINT,
    B8G8R8A8_SRGB =                                  VK_FORMAT_B8G8R8A8_SRGB,
    A8B8G8R8_UNORM_PACK32 =                          VK_FORMAT_A8B8G8R8_UNORM_PACK32,
    A8B8G8R8_SNORM_PACK32 =                          VK_FORMAT_A8B8G8R8_SNORM_PACK32,
    A8B8G8R8_USCALED_PACK32 =                        VK_FORMAT_A8B8G8R8_USCALED_PACK32,
    A8B8G8R8_SSCALED_PACK32 =                        VK_FORMAT_A8B8G8R8_SSCALED_PACK32,
    A8B8G8R8_UINT_PACK32 =                           VK_FORMAT_A8B8G8R8_UINT_PACK32,
    A8B8G8R8_SINT_PACK32 =                           VK_FORMAT_A8B8G8R8_SINT_PACK32,
    A8B8G8R8_SRGB_PACK32 =                           VK_FORMAT_A8B8G8R8_SRGB_PACK32,
    A2R10G10B10_UNORM_PACK32 =                       VK_FORMAT_A2R10G10B10_UNORM_PACK32,
    A2R10G10B10_SNORM_PACK32 =                       VK_FORMAT_A2R10G10B10_SNORM_PACK32,
    A2R10G10B10_USCALED_PACK32 =                     VK_FORMAT_A2R10G10B10_USCALED_PACK32,
    A2R10G10B10_SSCALED_PACK32 =                     VK_FORMAT_A2R10G10B10_SSCALED_PACK32,
    A2R10G10B10_UINT_PACK32 =                        VK_FORMAT_A2R10G10B10_UINT_PACK32,
    A2R10G10B10_SINT_PACK32 =                        VK_FORMAT_A2R10G10B10_SINT_PACK32,
    A2B10G10R10_UNORM_PACK32 =                       VK_FORMAT_A2B10G10R10_UNORM_PACK32,
    A2B10G10R10_SNORM_PACK32 =                       VK_FORMAT_A2B10G10R10_SNORM_PACK32,
    A2B10G10R10_USCALED_PACK32 =                     VK_FORMAT_A2B10G10R10_USCALED_PACK32,
    A2B10G10R10_SSCALED_PACK32 =                     VK_FORMAT_A2B10G10R10_SSCALED_PACK32,
    A2B10G10R10_UINT_PACK32 =                        VK_FORMAT_A2B10G10R10_UINT_PACK32,
    A2B10G10R10_SINT_PACK32 =                        VK_FORMAT_A2B10G10R10_SINT_PACK32,
    R16_UNORM =                                      VK_FORMAT_R16_UNORM,
    R16_SNORM =                                      VK_FORMAT_R16_SNORM,
    R16_USCALED =                                    VK_FORMAT_R16_USCALED,
    R16_SSCALED =                                    VK_FORMAT_R16_SSCALED,
    R16_UINT =                                       VK_FORMAT_R16_UINT,
    R16_SINT =                                       VK_FORMAT_R16_SINT,
    R16_SFLOAT =                                     VK_FORMAT_R16_SFLOAT,
    R16G16_UNORM =                                   VK_FORMAT_R16G16_UNORM,
    R16G16_SNORM =                                   VK_FORMAT_R16G16_SNORM,
    R16G16_USCALED =                                 VK_FORMAT_R16G16_USCALED,
    R16G16_SSCALED =                                 VK_FORMAT_R16G16_SSCALED,
    R16G16_UINT =                                    VK_FORMAT_R16G16_UINT,
    R16G16_SINT =                                    VK_FORMAT_R16G16_SINT,
    R16G16_SFLOAT =                                  VK_FORMAT_R16G16_SFLOAT,
    R16G16B16_UNORM =                                VK_FORMAT_R16G16B16_UNORM,
    R16G16B16_SNORM =                                VK_FORMAT_R16G16B16_SNORM,
    R16G16B16_USCALED =                              VK_FORMAT_R16G16B16_USCALED,
    R16G16B16_SSCALED =                              VK_FORMAT_R16G16B16_SSCALED,
    R16G16B16_UINT =                                 VK_FORMAT_R16G16B16_UINT,
    R16G16B16_SINT =                                 VK_FORMAT_R16G16B16_SINT,
    R16G16B16_SFLOAT =                               VK_FORMAT_R16G16B16_SFLOAT,
    R16G16B16A16_UNORM =                             VK_FORMAT_R16G16B16A16_UNORM,
    R16G16B16A16_SNORM =                             VK_FORMAT_R16G16B16A16_SNORM,
    R16G16B16A16_USCALED =                           VK_FORMAT_R16G16B16A16_USCALED,
    R16G16B16A16_SSCALED =                           VK_FORMAT_R16G16B16A16_SSCALED,
    R16G16B16A16_UINT =                              VK_FORMAT_R16G16B16A16_UINT,
    R16G16B16A16_SINT =                              VK_FORMAT_R16G16B16A16_SINT,
    R16G16B16A16_SFLOAT =                            VK_FORMAT_R16G16B16A16_SFLOAT,
    R32_UINT =                                       VK_FORMAT_R32_UINT,
    R32_SINT =                                       VK_FORMAT_R32_SINT,
    R32_SFLOAT =                                     VK_FORMAT_R32_SFLOAT,
    R32G32_UINT =                                    VK_FORMAT_R32G32_UINT,
    R32G32_SINT =                                    VK_FORMAT_R32G32_SINT,
    R32G32_SFLOAT =                                  VK_FORMAT_R32G32_SFLOAT,
    R32G32B32_UINT =                                 VK_FORMAT_R32G32B32_UINT,
    R32G32B32_SINT =                                 VK_FORMAT_R32G32B32_SINT,
    R32G32B32_SFLOAT =                               VK_FORMAT_R32G32B32_SFLOAT,
    R32G32B32A32_UINT =                              VK_FORMAT_R32G32B32A32_UINT,
    R32G32B32A32_SINT =                              VK_FORMAT_R32G32B32A32_SINT,
    R32G32B32A32_SFLOAT =                            VK_FORMAT_R32G32B32A32_SFLOAT,
    R64_UINT =                                       VK_FORMAT_R64_UINT,
    R64_SINT =                                       VK_FORMAT_R64_SINT,
    R64_SFLOAT =                                     VK_FORMAT_R64_SFLOAT,
    R64G64_UINT =                                    VK_FORMAT_R64G64_UINT,
    R64G64_SINT =                                    VK_FORMAT_R64G64_SINT,
    R64G64_SFLOAT =                                  VK_FORMAT_R64G64_SFLOAT,
    R64G64B64_UINT =                                 VK_FORMAT_R64G64B64_UINT,
    R64G64B64_SINT =                                 VK_FORMAT_R64G64B64_SINT,
    R64G64B64_SFLOAT =                               VK_FORMAT_R64G64B64_SFLOAT,
    R64G64B64A64_UINT =                              VK_FORMAT_R64G64B64A64_UINT,
    R64G64B64A64_SINT =                              VK_FORMAT_R64G64B64A64_SINT,
    R64G64B64A64_SFLOAT =                            VK_FORMAT_R64G64B64A64_SFLOAT,
    B10G11R11_UFLOAT_PACK32 =                        VK_FORMAT_B10G11R11_UFLOAT_PACK32,
    E5B9G9R9_UFLOAT_PACK32 =                         VK_FORMAT_E5B9G9R9_UFLOAT_PACK32,
    D16_UNORM =                                      VK_FORMAT_D16_UNORM,
    X8_D24_UNORM_PACK32 =                            VK_FORMAT_X8_D24_UNORM_PACK32,
    D32_SFLOAT =                                     VK_FORMAT_D32_SFLOAT,
    S8_UINT =                                        VK_FORMAT_S8_UINT,
    D16_UNORM_S8_UINT =                              VK_FORMAT_D16_UNORM_S8_UINT,
    D24_UNORM_S8_UINT =                              VK_FORMAT_D24_UNORM_S8_UINT,
    D32_SFLOAT_S8_UINT =                             VK_FORMAT_D32_SFLOAT_S8_UINT,
    BC1_RGB_UNORM_BLOCK =                            VK_FORMAT_BC1_RGB_UNORM_BLOCK,
    BC1_RGB_SRGB_BLOCK =                             VK_FORMAT_BC1_RGB_SRGB_BLOCK,
    BC1_RGBA_UNORM_BLOCK =                           VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
    BC1_RGBA_SRGB_BLOCK =                            VK_FORMAT_BC1_RGBA_SRGB_BLOCK,
    BC2_UNORM_BLOCK =                                VK_FORMAT_BC2_UNORM_BLOCK,
    BC2_SRGB_BLOCK =                                 VK_FORMAT_BC2_SRGB_BLOCK,
    BC3_UNORM_BLOCK =                                VK_FORMAT_BC3_UNORM_BLOCK,
    BC3_SRGB_BLOCK =                                 VK_FORMAT_BC3_SRGB_BLOCK,
    BC4_UNORM_BLOCK =                                VK_FORMAT_BC4_UNORM_BLOCK,
    BC4_SNORM_BLOCK =                                VK_FORMAT_BC4_SNORM_BLOCK,
    BC5_UNORM_BLOCK =                                VK_FORMAT_BC5_UNORM_BLOCK,
    BC5_SNORM_BLOCK =                                VK_FORMAT_BC5_SNORM_BLOCK,
    BC6H_UFLOAT_BLOCK =                              VK_FORMAT_BC6H_UFLOAT_BLOCK,
    BC6H_SFLOAT_BLOCK =                              VK_FORMAT_BC6H_SFLOAT_BLOCK,
    BC7_UNORM_BLOCK =                                VK_FORMAT_BC7_UNORM_BLOCK,
    BC7_SRGB_BLOCK =                                 VK_FORMAT_BC7_SRGB_BLOCK,
    ETC2_R8G8B8_UNORM_BLOCK =                        VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK,
    ETC2_R8G8B8_SRGB_BLOCK =                         VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK,
    ETC2_R8G8B8A1_UNORM_BLOCK =                      VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK,
    ETC2_R8G8B8A1_SRGB_BLOCK =                       VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK,
    ETC2_R8G8B8A8_UNORM_BLOCK =                      VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK,
    ETC2_R8G8B8A8_SRGB_BLOCK =                       VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK,
    EAC_R11_UNORM_BLOCK =                            VK_FORMAT_EAC_R11_UNORM_BLOCK,
    EAC_R11_SNORM_BLOCK =                            VK_FORMAT_EAC_R11_SNORM_BLOCK,
    EAC_R11G11_UNORM_BLOCK =                         VK_FORMAT_EAC_R11G11_UNORM_BLOCK,
    EAC_R11G11_SNORM_BLOCK =                         VK_FORMAT_EAC_R11G11_SNORM_BLOCK,
    ASTC_4x4_UNORM_BLOCK =                           VK_FORMAT_ASTC_4x4_UNORM_BLOCK,
    ASTC_4x4_SRGB_BLOCK =                            VK_FORMAT_ASTC_4x4_SRGB_BLOCK,
    ASTC_5x4_UNORM_BLOCK =                           VK_FORMAT_ASTC_5x4_UNORM_BLOCK,
    ASTC_5x4_SRGB_BLOCK =                            VK_FORMAT_ASTC_5x4_SRGB_BLOCK,
    ASTC_5x5_UNORM_BLOCK =                           VK_FORMAT_ASTC_5x5_UNORM_BLOCK,
    ASTC_5x5_SRGB_BLOCK =                            VK_FORMAT_ASTC_5x5_SRGB_BLOCK,
    ASTC_6x5_UNORM_BLOCK =                           VK_FORMAT_ASTC_6x5_UNORM_BLOCK,
    ASTC_6x5_SRGB_BLOCK =                            VK_FORMAT_ASTC_6x5_SRGB_BLOCK,
    ASTC_6x6_UNORM_BLOCK =                           VK_FORMAT_ASTC_6x6_UNORM_BLOCK,
    ASTC_6x6_SRGB_BLOCK =                            VK_FORMAT_ASTC_6x6_SRGB_BLOCK,
    ASTC_8x5_UNORM_BLOCK =                           VK_FORMAT_ASTC_8x5_UNORM_BLOCK,
    ASTC_8x5_SRGB_BLOCK =                            VK_FORMAT_ASTC_8x5_SRGB_BLOCK,
    ASTC_8x6_UNORM_BLOCK =                           VK_FORMAT_ASTC_8x6_UNORM_BLOCK,
    ASTC_8x6_SRGB_BLOCK =                            VK_FORMAT_ASTC_8x6_SRGB_BLOCK,
    ASTC_8x8_UNORM_BLOCK =                           VK_FORMAT_ASTC_8x8_UNORM_BLOCK,
    ASTC_8x8_SRGB_BLOCK =                            VK_FORMAT_ASTC_8x8_SRGB_BLOCK,
    ASTC_10x5_UNORM_BLOCK =                          VK_FORMAT_ASTC_10x5_UNORM_BLOCK,
    ASTC_10x5_SRGB_BLOCK =                           VK_FORMAT_ASTC_10x5_SRGB_BLOCK,
    ASTC_10x6_UNORM_BLOCK =                          VK_FORMAT_ASTC_10x6_UNORM_BLOCK,
    ASTC_10x6_SRGB_BLOCK =                           VK_FORMAT_ASTC_10x6_SRGB_BLOCK,
    ASTC_10x8_UNORM_BLOCK =                          VK_FORMAT_ASTC_10x8_UNORM_BLOCK,
    ASTC_10x8_SRGB_BLOCK =                           VK_FORMAT_ASTC_10x8_SRGB_BLOCK,
    ASTC_10x10_UNORM_BLOCK =                         VK_FORMAT_ASTC_10x10_UNORM_BLOCK,
    ASTC_10x10_SRGB_BLOCK =                          VK_FORMAT_ASTC_10x10_SRGB_BLOCK,
    ASTC_12x10_UNORM_BLOCK =                         VK_FORMAT_ASTC_12x10_UNORM_BLOCK,
    ASTC_12x10_SRGB_BLOCK =                          VK_FORMAT_ASTC_12x10_SRGB_BLOCK,
    ASTC_12x12_UNORM_BLOCK =                         VK_FORMAT_ASTC_12x12_UNORM_BLOCK,
    ASTC_12x12_SRGB_BLOCK =                          VK_FORMAT_ASTC_12x12_SRGB_BLOCK,
    G8B8G8R8_422_UNORM =                             VK_FORMAT_G8B8G8R8_422_UNORM,
    B8G8R8G8_422_UNORM =                             VK_FORMAT_B8G8R8G8_422_UNORM,
    G8_B8_R8_3PLANE_420_UNORM =                      VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM,
    G8_B8R8_2PLANE_420_UNORM =                       VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
    G8_B8_R8_3PLANE_422_UNORM =                      VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM,
    G8_B8R8_2PLANE_422_UNORM =                       VK_FORMAT_G8_B8R8_2PLANE_422_UNORM,
    G8_B8_R8_3PLANE_444_UNORM =                      VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM,
    R10X6_UNORM_PACK16 =                             VK_FORMAT_R10X6_UNORM_PACK16,
    R10X6G10X6_UNORM_2PACK16 =                       VK_FORMAT_R10X6G10X6_UNORM_2PACK16,
    R10X6G10X6B10X6A10X6_UNORM_4PACK16 =             VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16,
    G10X6B10X6G10X6R10X6_422_UNORM_4PACK16 =         VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16,
    B10X6G10X6R10X6G10X6_422_UNORM_4PACK16 =         VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16,
    G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16 =     VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16,
    G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 =      VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
    G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16 =     VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16,
    G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16 =      VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16,
    G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16 =     VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16,
    R12X4_UNORM_PACK16 =                             VK_FORMAT_R12X4_UNORM_PACK16,
    R12X4G12X4_UNORM_2PACK16 =                       VK_FORMAT_R12X4G12X4_UNORM_2PACK16,
    R12X4G12X4B12X4A12X4_UNORM_4PACK16 =             VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16,
    G12X4B12X4G12X4R12X4_422_UNORM_4PACK16 =         VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16,
    B12X4G12X4R12X4G12X4_422_UNORM_4PACK16 =         VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16,
    G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16 =     VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16,
    G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16 =      VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16,
    G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16 =     VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16,
    G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16 =      VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16,
    G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16 =     VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16,
    G16B16G16R16_422_UNORM =                         VK_FORMAT_G16B16G16R16_422_UNORM,
    B16G16R16G16_422_UNORM =                         VK_FORMAT_B16G16R16G16_422_UNORM,
    G16_B16_R16_3PLANE_420_UNORM =                   VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM,
    G16_B16R16_2PLANE_420_UNORM =                    VK_FORMAT_G16_B16R16_2PLANE_420_UNORM,
    G16_B16_R16_3PLANE_422_UNORM =                   VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM,
    G16_B16R16_2PLANE_422_UNORM =                    VK_FORMAT_G16_B16R16_2PLANE_422_UNORM,
    G16_B16_R16_3PLANE_444_UNORM =                   VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM,
    G8_B8R8_2PLANE_444_UNORM =                       VK_FORMAT_G8_B8R8_2PLANE_444_UNORM,
    G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16 =      VK_FORMAT_G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16,
    G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16 =      VK_FORMAT_G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16,
    G16_B16R16_2PLANE_444_UNORM =                    VK_FORMAT_G16_B16R16_2PLANE_444_UNORM,
    A4R4G4B4_UNORM_PACK16 =                          VK_FORMAT_A4R4G4B4_UNORM_PACK16,
    A4B4G4R4_UNORM_PACK16 =                          VK_FORMAT_A4B4G4R4_UNORM_PACK16,
    ASTC_4x4_SFLOAT_BLOCK =                          VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK,
    ASTC_5x4_SFLOAT_BLOCK =                          VK_FORMAT_ASTC_5x4_SFLOAT_BLOCK,
    ASTC_5x5_SFLOAT_BLOCK =                          VK_FORMAT_ASTC_5x5_SFLOAT_BLOCK,
    ASTC_6x5_SFLOAT_BLOCK =                          VK_FORMAT_ASTC_6x5_SFLOAT_BLOCK,
    ASTC_6x6_SFLOAT_BLOCK =                          VK_FORMAT_ASTC_6x6_SFLOAT_BLOCK,
    ASTC_8x5_SFLOAT_BLOCK =                          VK_FORMAT_ASTC_8x5_SFLOAT_BLOCK,
    ASTC_8x6_SFLOAT_BLOCK =                          VK_FORMAT_ASTC_8x6_SFLOAT_BLOCK,
    ASTC_8x8_SFLOAT_BLOCK =                          VK_FORMAT_ASTC_8x8_SFLOAT_BLOCK,
    ASTC_10x5_SFLOAT_BLOCK =                         VK_FORMAT_ASTC_10x5_SFLOAT_BLOCK,
    ASTC_10x6_SFLOAT_BLOCK =                         VK_FORMAT_ASTC_10x6_SFLOAT_BLOCK,
    ASTC_10x8_SFLOAT_BLOCK =                         VK_FORMAT_ASTC_10x8_SFLOAT_BLOCK,
    ASTC_10x10_SFLOAT_BLOCK =                        VK_FORMAT_ASTC_10x10_SFLOAT_BLOCK,
    ASTC_12x10_SFLOAT_BLOCK =                        VK_FORMAT_ASTC_12x10_SFLOAT_BLOCK,
    ASTC_12x12_SFLOAT_BLOCK =                        VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK,
    PVRTC1_2BPP_UNORM_BLOCK_IMG =                    VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG,
    PVRTC1_4BPP_UNORM_BLOCK_IMG =                    VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG,
    PVRTC2_2BPP_UNORM_BLOCK_IMG =                    VK_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG,
    PVRTC2_4BPP_UNORM_BLOCK_IMG =                    VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG,
    PVRTC1_2BPP_SRGB_BLOCK_IMG =                     VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG,
    PVRTC1_4BPP_SRGB_BLOCK_IMG =                     VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG,
    PVRTC2_2BPP_SRGB_BLOCK_IMG =                     VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG,
    PVRTC2_4BPP_SRGB_BLOCK_IMG =                     VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG,
    R16G16_SFIXED5_NV =                              VK_FORMAT_R16G16_SFIXED5_NV,
    A1B5G5R5_UNORM_PACK16_KHR =                      VK_FORMAT_A1B5G5R5_UNORM_PACK16_KHR,
    A8_UNORM_KHR =                                   VK_FORMAT_A8_UNORM_KHR,
    ASTC_4x4_SFLOAT_BLOCK_EXT =                      VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK_EXT,
    ASTC_5x4_SFLOAT_BLOCK_EXT =                      VK_FORMAT_ASTC_5x4_SFLOAT_BLOCK_EXT,
    ASTC_5x5_SFLOAT_BLOCK_EXT =                      VK_FORMAT_ASTC_5x5_SFLOAT_BLOCK_EXT,
    ASTC_6x5_SFLOAT_BLOCK_EXT =                      VK_FORMAT_ASTC_6x5_SFLOAT_BLOCK_EXT,
    ASTC_6x6_SFLOAT_BLOCK_EXT =                      VK_FORMAT_ASTC_6x6_SFLOAT_BLOCK_EXT,
    ASTC_8x5_SFLOAT_BLOCK_EXT =                      VK_FORMAT_ASTC_8x5_SFLOAT_BLOCK_EXT,
    ASTC_8x6_SFLOAT_BLOCK_EXT =                      VK_FORMAT_ASTC_8x6_SFLOAT_BLOCK_EXT,
    ASTC_8x8_SFLOAT_BLOCK_EXT =                      VK_FORMAT_ASTC_8x8_SFLOAT_BLOCK_EXT,
    ASTC_10x5_SFLOAT_BLOCK_EXT =                     VK_FORMAT_ASTC_10x5_SFLOAT_BLOCK_EXT,
    ASTC_10x6_SFLOAT_BLOCK_EXT =                     VK_FORMAT_ASTC_10x6_SFLOAT_BLOCK_EXT,
    ASTC_10x8_SFLOAT_BLOCK_EXT =                     VK_FORMAT_ASTC_10x8_SFLOAT_BLOCK_EXT,
    ASTC_10x10_SFLOAT_BLOCK_EXT =                    VK_FORMAT_ASTC_10x10_SFLOAT_BLOCK_EXT,
    ASTC_12x10_SFLOAT_BLOCK_EXT =                    VK_FORMAT_ASTC_12x10_SFLOAT_BLOCK_EXT,
    ASTC_12x12_SFLOAT_BLOCK_EXT =                    VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK_EXT,
    G8B8G8R8_422_UNORM_KHR =                         VK_FORMAT_G8B8G8R8_422_UNORM_KHR,
    B8G8R8G8_422_UNORM_KHR =                         VK_FORMAT_B8G8R8G8_422_UNORM_KHR,
    G8_B8_R8_3PLANE_420_UNORM_KHR =                  VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM_KHR,
    G8_B8R8_2PLANE_420_UNORM_KHR =                   VK_FORMAT_G8_B8R8_2PLANE_420_UNORM_KHR,
    G8_B8_R8_3PLANE_422_UNORM_KHR =                  VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM_KHR,
    G8_B8R8_2PLANE_422_UNORM_KHR =                   VK_FORMAT_G8_B8R8_2PLANE_422_UNORM_KHR,
    G8_B8_R8_3PLANE_444_UNORM_KHR =                  VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM_KHR,
    R10X6_UNORM_PACK16_KHR =                         VK_FORMAT_R10X6_UNORM_PACK16_KHR,
    R10X6G10X6_UNORM_2PACK16_KHR =                   VK_FORMAT_R10X6G10X6_UNORM_2PACK16_KHR,
    R10X6G10X6B10X6A10X6_UNORM_4PACK16_KHR =         VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16_KHR,
    G10X6B10X6G10X6R10X6_422_UNORM_4PACK16_KHR =     VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16_KHR,
    B10X6G10X6R10X6G10X6_422_UNORM_4PACK16_KHR =     VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16_KHR,
    G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16_KHR = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16_KHR,
    G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16_KHR =  VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16_KHR,
    G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16_KHR = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16_KHR,
    G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16_KHR =  VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16_KHR,
    G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16_KHR = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16_KHR,
    R12X4_UNORM_PACK16_KHR =                         VK_FORMAT_R12X4_UNORM_PACK16_KHR,
    R12X4G12X4_UNORM_2PACK16_KHR =                   VK_FORMAT_R12X4G12X4_UNORM_2PACK16_KHR,
    R12X4G12X4B12X4A12X4_UNORM_4PACK16_KHR =         VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16_KHR,
    G12X4B12X4G12X4R12X4_422_UNORM_4PACK16_KHR =     VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16_KHR,
    B12X4G12X4R12X4G12X4_422_UNORM_4PACK16_KHR =     VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16_KHR,
    G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16_KHR = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16_KHR,
    G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16_KHR =  VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16_KHR,
    G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16_KHR = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16_KHR,
    G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16_KHR =  VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16_KHR,
    G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16_KHR = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16_KHR,
    G16B16G16R16_422_UNORM_KHR =                     VK_FORMAT_G16B16G16R16_422_UNORM_KHR,
    B16G16R16G16_422_UNORM_KHR =                     VK_FORMAT_B16G16R16G16_422_UNORM_KHR,
    G16_B16_R16_3PLANE_420_UNORM_KHR =               VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM_KHR,
    G16_B16R16_2PLANE_420_UNORM_KHR =                VK_FORMAT_G16_B16R16_2PLANE_420_UNORM_KHR,
    G16_B16_R16_3PLANE_422_UNORM_KHR =               VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM_KHR,
    G16_B16R16_2PLANE_422_UNORM_KHR =                VK_FORMAT_G16_B16R16_2PLANE_422_UNORM_KHR,
    G16_B16_R16_3PLANE_444_UNORM_KHR =               VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM_KHR,
    G8_B8R8_2PLANE_444_UNORM_EXT =                   VK_FORMAT_G8_B8R8_2PLANE_444_UNORM_EXT,
    G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16_EXT =  VK_FORMAT_G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16_EXT,
    G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16_EXT =  VK_FORMAT_G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16_EXT,
    G16_B16R16_2PLANE_444_UNORM_EXT =                VK_FORMAT_G16_B16R16_2PLANE_444_UNORM_EXT,
    A4R4G4B4_UNORM_PACK16_EXT =                      VK_FORMAT_A4R4G4B4_UNORM_PACK16_EXT,
    A4B4G4R4_UNORM_PACK16_EXT =                      VK_FORMAT_A4B4G4R4_UNORM_PACK16_EXT,
    R16G16_S10_5_NV =                                VK_FORMAT_R16G16_S10_5_NV,
};

struct VertexAttribute: gfx::VertexAttributeDesc {
    VertexAttribute(u32 location, u32 binding, Format format, u32 offset)
        : gfx::VertexAttributeDesc {
              .location = location,
              .binding = binding,
              .format = (VkFormat)format,
              .offset = offset,
          }
    {
    }
};
static_assert(sizeof(VertexAttribute) == sizeof(gfx::VertexAttributeDesc));

enum class PrimitiveTopology {
    PointList = VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
    LineList = VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
    LineStrip = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP,
    TriangleList = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    TriangleStrip = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
    TriangleFan = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,
    LineListWithAdjacency = VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY,
    LineStripWithAdjacency = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY,
    TriangleListWithAdjacency = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY,
    TriangleStripWithAdjacency = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY,
    PatchList = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST,
};

struct InputAssembly: gfx::InputAssemblyDesc{
    InputAssembly(PrimitiveTopology primitive_topology = PrimitiveTopology::TriangleList, bool primitive_restart_enable = false)
        : gfx::InputAssemblyDesc {
              .primitive_topology = (VkPrimitiveTopology)primitive_topology,
              .primitive_restart_enable = primitive_restart_enable,
          }
    {
    }
};
static_assert(sizeof(InputAssembly) == sizeof(gfx::InputAssemblyDesc));


enum class BlendFactor {

};

struct Attachment: gfx::AttachmentDesc {
    Attachment(
        VkFormat format,
        VkBool32 blend_enable = false,
        VkBlendFactor src_color_blend_factor = VK_BLEND_FACTOR_ZERO,
        VkBlendFactor dst_color_blend_factor = VK_BLEND_FACTOR_ZERO,
        VkBlendOp color_blend_op = VK_BLEND_OP_ADD,
        VkBlendFactor src_alpha_blend_factor = VK_BLEND_FACTOR_ZERO,
        VkBlendFactor dst_alpha_blend_factor = VK_BLEND_FACTOR_ZERO,
        VkBlendOp alpha_blend_op = VK_BLEND_OP_ADD,
        VkColorComponentFlags color_write_mask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
    )
        : gfx::AttachmentDesc {
          }
    {
    }
};

struct GraphicsPipeline: nb::intrusive_base {
    GraphicsPipeline(nb::ref<Context> ctx,
        const std::vector<nb::ref<PipelineStage>>& stages,
        const std::vector<VertexBinding>& vertex_bindings,
        const std::vector<VertexAttribute>& vertex_attributes,
        InputAssembly input_assembly,
        const std::vector<nb::ref<DescriptorSet>>& descriptor_sets
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
            .descriptor_sets = ArrayView(d),
        });

        if (vkr != VK_SUCCESS) {
            throw std::exception("Failed to create graphics pipeline");
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

struct CommandPool {
    CommandPool(VkCommandPool obj): obj(obj) {}

    VkCommandPool obj;
};

struct CommandBuffer {
    CommandBuffer(VkCommandBuffer obj): obj(obj) {}

    VkCommandBuffer obj;
};

void BeginCommands(const CommandPool& command_pool, const CommandBuffer& command_buffer, const Context& ctx) {
    VkResult vkr = gfx::BeginCommands(command_pool.obj, command_buffer.obj, ctx.vk);
    if(vkr != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin commands");
    }
}

void EndCommands(const CommandBuffer& command_buffer) {
    VkResult vkr = gfx::EndCommands(command_buffer.obj);
    if(vkr != VK_SUCCESS) {
        throw std::runtime_error("Failed to end commands");
    }
}

typedef ImU32 Color;

ImGuiStyle& ImGui_get_style() {
    ImGui::CreateContext();
    return ImGui::GetStyle();
}

nb::object test(nb::callable callable) {
    return callable();
}

bool test2(std::function<bool()> callable) {
    return callable();
}

#if 0
struct SlangType {
    slang::TypeReflection::Kind kind;

    // Layout - common
    SlangParameterCategory category;
    usize size;
    usize alignment;
    usize stride; // ALIGN_UP(size, alignment)
};

struct SlangLayout {

};

struct SlangVariable {
    std::string name;
    SlangType type;
    SlangLayout layout;
};

struct SlangType_Struct {
    std::vector<SlangVariable> fields;
};

struct SlangType_Array {
    SlangType element_type;
    usize count;
};

struct SlangType_Vector {
    SlangType element_type;
    u32 count;
};

struct SlangType_Matrix {
    SlangType element_type;
    u32 rows;
    u32 columns;
};

struct SlangType_Resource {
    SlangResourceShape shape;
    SlangResourceAccess access;
    SlangType type;
};

struct SlangType_Container {
    // Content
    SlangType element_type;
};

struct Program {
    // Global scope

    // Entry point
};
#endif

NB_MODULE(pyxpg, m) {
    nb::intrusive_init(
        [](PyObject *o) noexcept {
            nb::gil_scoped_acquire guard;
            Py_INCREF(o);
        },
        [](PyObject *o) noexcept {
            nb::gil_scoped_acquire guard;
            Py_DECREF(o);
        });

    nb::class_<Context>(m, "Context",
        nb::intrusive_ptr<Context>([](Context *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<>())
    ;

    nb::class_<Window>(m, "Window",
        nb::type_slots(Window::tp_slots),
        nb::intrusive_ptr<Window>([](Window *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<nb::ref<Context>, const::std::string&, u32, u32>(), nb::arg("ctx"), nb::arg("title"), nb::arg("width"), nb::arg("height"))
        .def("should_close", &Window::should_close)
        .def("set_callbacks", &Window::set_callbacks, nb::arg("draw"))
        .def("reset_callbacks", &Window::reset_callbacks)
        .def("update_swapchain", &Window::update_swapchain)
        .def("begin_frame", &Window::begin_frame)
        .def("end_frame", &Window::end_frame, nb::arg("frame"))
    ;

    nb::class_<Gui>(m, "Gui",
        nb::type_slots(Gui::tp_slots),
        nb::intrusive_ptr<Gui>([](Gui *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<nb::ref<Window>>(), nb::arg("window"))
        .def("begin_frame", &Gui::begin_frame)
        .def("end_frame", &Gui::end_frame)
        .def("render", &Gui::render, nb::arg("frame"))
    ;

    nb::enum_<gfx::AllocPresets::Type>(m, "AllocType")
        .value("HOST", gfx::AllocPresets::Type::Host)
        .value("HOST_WRITE_COMBINING", gfx::AllocPresets::Type::HostWriteCombining)
        .value("DEVICE_MAPPED_WITH_FALLBACK", gfx::AllocPresets::Type::DeviceMappedWithFallback)
        .value("DEVICE", gfx::AllocPresets::Type::Device)
        .value("DEVICE_DEDICATED", gfx::AllocPresets::Type::DeviceDedicated)
    ;

    nb::enum_<Buffer::UsageFlags>(m, "BufferUsageFlags", nb::is_arithmetic() , nb::is_flag())
        .value("TRANSFER_SRC", Buffer::UsageFlags::TransferSrc)
        .value("TRANSFER_DST", Buffer::UsageFlags::TransferDst)
        .value("UNIFORM", Buffer::UsageFlags::Uniform)
        .value("STORAGE", Buffer::UsageFlags::Storage)
        .value("INDEX", Buffer::UsageFlags::Index)
        .value("VERTEX", Buffer::UsageFlags::Vertex)
        .value("INDIRECT", Buffer::UsageFlags::Indirect)
        .value("ACCELERATION_STRUCTURE_INPUT", Buffer::UsageFlags::AccelerationStructureInput)
        .value("ACCELERATION_STRUCTURE_STORAGE", Buffer::UsageFlags::AccelerationStructureStorage)
    ;

    nb::class_<Buffer>(m, "Buffer",
        nb::intrusive_ptr<Buffer>([](Buffer *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<nb::ref<Context>, size_t, Buffer::UsageFlags, gfx::AllocPresets::Type>(), nb::arg("ctx"), nb::arg("size"), nb::arg("usage_flags"), nb::arg("alloc_type"))
        .def(nb::init<nb::ref<Context>, nb::bytes, Buffer::UsageFlags, gfx::AllocPresets::Type>(), nb::arg("ctx"), nb::arg("data"), nb::arg("usage_flags"), nb::arg("alloc_type"))
        .def("destroy", &Buffer::destroy)
    ;

    nb::class_<Shader>(m, "Shader",
        nb::intrusive_ptr<Shader>([](Shader *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<nb::ref<Context>, nb::bytes>(), nb::arg("ctx"), nb::arg("code"))
        .def("destroy", &Shader::destroy)
    ;


    nb::enum_<PipelineStage::Stage>(m, "Stage")
        .value("VERTEX", PipelineStage::Stage::Vertex)
        .value("TESSELLATION_CONTROL", PipelineStage::Stage::TessellationControl)
        .value("TESSELLATION_EVALUATION", PipelineStage::Stage::TessellationEvaluation)
        .value("GEOMETRY", PipelineStage::Stage::Geometry)
        .value("FRAGMENT", PipelineStage::Stage::Fragment)
        .value("COMPUTE", PipelineStage::Stage::Compute)
        .value("RAYGEN", PipelineStage::Stage::Raygen)
        .value("ANY_HIT", PipelineStage::Stage::AnyHit)
        .value("CLOSEST_HIT", PipelineStage::Stage::ClosestHit)
        .value("MISS", PipelineStage::Stage::Miss)
        .value("INTERSECTION", PipelineStage::Stage::Intersection)
        .value("CALLABLE", PipelineStage::Stage::Callable)
        .value("TASK_EXT", PipelineStage::Stage::Task)
        .value("MESH_EXT", PipelineStage::Stage::Mesh)
    ;

    nb::class_<PipelineStage>(m, "PipelineStage",
        nb::intrusive_ptr<PipelineStage>([](PipelineStage *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<nb::ref<Shader>, PipelineStage::Stage, std::string>(), nb::arg("shader"), nb::arg("stage"), nb::arg("entry") = "main")
    ;

    nb::enum_<VertexBinding::InputRate>(m, "VertexInputRate")
        .value("VERTEX", VertexBinding::InputRate::Vertex)
        .value("INSTANCE", VertexBinding::InputRate::Instance)
    ;

    nb::class_<VertexBinding>(m, "VertexBinding")
        .def(nb::init<u32, u32, VertexBinding::InputRate>(), nb::arg("binding"), nb::arg("stride"), nb::arg("input_rate") = VertexBinding::InputRate::Vertex)
    ;

    nb::enum_<Format>(m, "Format")
        .value("UNDEFINED", Format::UNDEFINED)
        .value("R4G4_UNORM_PACK8", Format::R4G4_UNORM_PACK8)
        .value("R4G4B4A4_UNORM_PACK16", Format::R4G4B4A4_UNORM_PACK16)
        .value("B4G4R4A4_UNORM_PACK16", Format::B4G4R4A4_UNORM_PACK16)
        .value("R5G6B5_UNORM_PACK16", Format::R5G6B5_UNORM_PACK16)
        .value("B5G6R5_UNORM_PACK16", Format::B5G6R5_UNORM_PACK16)
        .value("R5G5B5A1_UNORM_PACK16", Format::R5G5B5A1_UNORM_PACK16)
        .value("B5G5R5A1_UNORM_PACK16", Format::B5G5R5A1_UNORM_PACK16)
        .value("A1R5G5B5_UNORM_PACK16", Format::A1R5G5B5_UNORM_PACK16)
        .value("R8_UNORM", Format::R8_UNORM)
        .value("R8_SNORM", Format::R8_SNORM)
        .value("R8_USCALED", Format::R8_USCALED)
        .value("R8_SSCALED", Format::R8_SSCALED)
        .value("R8_UINT", Format::R8_UINT)
        .value("R8_SINT", Format::R8_SINT)
        .value("R8_SRGB", Format::R8_SRGB)
        .value("R8G8_UNORM", Format::R8G8_UNORM)
        .value("R8G8_SNORM", Format::R8G8_SNORM)
        .value("R8G8_USCALED", Format::R8G8_USCALED)
        .value("R8G8_SSCALED", Format::R8G8_SSCALED)
        .value("R8G8_UINT", Format::R8G8_UINT)
        .value("R8G8_SINT", Format::R8G8_SINT)
        .value("R8G8_SRGB", Format::R8G8_SRGB)
        .value("R8G8B8_UNORM", Format::R8G8B8_UNORM)
        .value("R8G8B8_SNORM", Format::R8G8B8_SNORM)
        .value("R8G8B8_USCALED", Format::R8G8B8_USCALED)
        .value("R8G8B8_SSCALED", Format::R8G8B8_SSCALED)
        .value("R8G8B8_UINT", Format::R8G8B8_UINT)
        .value("R8G8B8_SINT", Format::R8G8B8_SINT)
        .value("R8G8B8_SRGB", Format::R8G8B8_SRGB)
        .value("B8G8R8_UNORM", Format::B8G8R8_UNORM)
        .value("B8G8R8_SNORM", Format::B8G8R8_SNORM)
        .value("B8G8R8_USCALED", Format::B8G8R8_USCALED)
        .value("B8G8R8_SSCALED", Format::B8G8R8_SSCALED)
        .value("B8G8R8_UINT", Format::B8G8R8_UINT)
        .value("B8G8R8_SINT", Format::B8G8R8_SINT)
        .value("B8G8R8_SRGB", Format::B8G8R8_SRGB)
        .value("R8G8B8A8_UNORM", Format::R8G8B8A8_UNORM)
        .value("R8G8B8A8_SNORM", Format::R8G8B8A8_SNORM)
        .value("R8G8B8A8_USCALED", Format::R8G8B8A8_USCALED)
        .value("R8G8B8A8_SSCALED", Format::R8G8B8A8_SSCALED)
        .value("R8G8B8A8_UINT", Format::R8G8B8A8_UINT)
        .value("R8G8B8A8_SINT", Format::R8G8B8A8_SINT)
        .value("R8G8B8A8_SRGB", Format::R8G8B8A8_SRGB)
        .value("B8G8R8A8_UNORM", Format::B8G8R8A8_UNORM)
        .value("B8G8R8A8_SNORM", Format::B8G8R8A8_SNORM)
        .value("B8G8R8A8_USCALED", Format::B8G8R8A8_USCALED)
        .value("B8G8R8A8_SSCALED", Format::B8G8R8A8_SSCALED)
        .value("B8G8R8A8_UINT", Format::B8G8R8A8_UINT)
        .value("B8G8R8A8_SINT", Format::B8G8R8A8_SINT)
        .value("B8G8R8A8_SRGB", Format::B8G8R8A8_SRGB)
        .value("A8B8G8R8_UNORM_PACK32", Format::A8B8G8R8_UNORM_PACK32)
        .value("A8B8G8R8_SNORM_PACK32", Format::A8B8G8R8_SNORM_PACK32)
        .value("A8B8G8R8_USCALED_PACK32", Format::A8B8G8R8_USCALED_PACK32)
        .value("A8B8G8R8_SSCALED_PACK32", Format::A8B8G8R8_SSCALED_PACK32)
        .value("A8B8G8R8_UINT_PACK32", Format::A8B8G8R8_UINT_PACK32)
        .value("A8B8G8R8_SINT_PACK32", Format::A8B8G8R8_SINT_PACK32)
        .value("A8B8G8R8_SRGB_PACK32", Format::A8B8G8R8_SRGB_PACK32)
        .value("A2R10G10B10_UNORM_PACK32", Format::A2R10G10B10_UNORM_PACK32)
        .value("A2R10G10B10_SNORM_PACK32", Format::A2R10G10B10_SNORM_PACK32)
        .value("A2R10G10B10_USCALED_PACK32", Format::A2R10G10B10_USCALED_PACK32)
        .value("A2R10G10B10_SSCALED_PACK32", Format::A2R10G10B10_SSCALED_PACK32)
        .value("A2R10G10B10_UINT_PACK32", Format::A2R10G10B10_UINT_PACK32)
        .value("A2R10G10B10_SINT_PACK32", Format::A2R10G10B10_SINT_PACK32)
        .value("A2B10G10R10_UNORM_PACK32", Format::A2B10G10R10_UNORM_PACK32)
        .value("A2B10G10R10_SNORM_PACK32", Format::A2B10G10R10_SNORM_PACK32)
        .value("A2B10G10R10_USCALED_PACK32", Format::A2B10G10R10_USCALED_PACK32)
        .value("A2B10G10R10_SSCALED_PACK32", Format::A2B10G10R10_SSCALED_PACK32)
        .value("A2B10G10R10_UINT_PACK32", Format::A2B10G10R10_UINT_PACK32)
        .value("A2B10G10R10_SINT_PACK32", Format::A2B10G10R10_SINT_PACK32)
        .value("R16_UNORM", Format::R16_UNORM)
        .value("R16_SNORM", Format::R16_SNORM)
        .value("R16_USCALED", Format::R16_USCALED)
        .value("R16_SSCALED", Format::R16_SSCALED)
        .value("R16_UINT", Format::R16_UINT)
        .value("R16_SINT", Format::R16_SINT)
        .value("R16_SFLOAT", Format::R16_SFLOAT)
        .value("R16G16_UNORM", Format::R16G16_UNORM)
        .value("R16G16_SNORM", Format::R16G16_SNORM)
        .value("R16G16_USCALED", Format::R16G16_USCALED)
        .value("R16G16_SSCALED", Format::R16G16_SSCALED)
        .value("R16G16_UINT", Format::R16G16_UINT)
        .value("R16G16_SINT", Format::R16G16_SINT)
        .value("R16G16_SFLOAT", Format::R16G16_SFLOAT)
        .value("R16G16B16_UNORM", Format::R16G16B16_UNORM)
        .value("R16G16B16_SNORM", Format::R16G16B16_SNORM)
        .value("R16G16B16_USCALED", Format::R16G16B16_USCALED)
        .value("R16G16B16_SSCALED", Format::R16G16B16_SSCALED)
        .value("R16G16B16_UINT", Format::R16G16B16_UINT)
        .value("R16G16B16_SINT", Format::R16G16B16_SINT)
        .value("R16G16B16_SFLOAT", Format::R16G16B16_SFLOAT)
        .value("R16G16B16A16_UNORM", Format::R16G16B16A16_UNORM)
        .value("R16G16B16A16_SNORM", Format::R16G16B16A16_SNORM)
        .value("R16G16B16A16_USCALED", Format::R16G16B16A16_USCALED)
        .value("R16G16B16A16_SSCALED", Format::R16G16B16A16_SSCALED)
        .value("R16G16B16A16_UINT", Format::R16G16B16A16_UINT)
        .value("R16G16B16A16_SINT", Format::R16G16B16A16_SINT)
        .value("R16G16B16A16_SFLOAT", Format::R16G16B16A16_SFLOAT)
        .value("R32_UINT", Format::R32_UINT)
        .value("R32_SINT", Format::R32_SINT)
        .value("R32_SFLOAT", Format::R32_SFLOAT)
        .value("R32G32_UINT", Format::R32G32_UINT)
        .value("R32G32_SINT", Format::R32G32_SINT)
        .value("R32G32_SFLOAT", Format::R32G32_SFLOAT)
        .value("R32G32B32_UINT", Format::R32G32B32_UINT)
        .value("R32G32B32_SINT", Format::R32G32B32_SINT)
        .value("R32G32B32_SFLOAT", Format::R32G32B32_SFLOAT)
        .value("R32G32B32A32_UINT", Format::R32G32B32A32_UINT)
        .value("R32G32B32A32_SINT", Format::R32G32B32A32_SINT)
        .value("R32G32B32A32_SFLOAT", Format::R32G32B32A32_SFLOAT)
        .value("R64_UINT", Format::R64_UINT)
        .value("R64_SINT", Format::R64_SINT)
        .value("R64_SFLOAT", Format::R64_SFLOAT)
        .value("R64G64_UINT", Format::R64G64_UINT)
        .value("R64G64_SINT", Format::R64G64_SINT)
        .value("R64G64_SFLOAT", Format::R64G64_SFLOAT)
        .value("R64G64B64_UINT", Format::R64G64B64_UINT)
        .value("R64G64B64_SINT", Format::R64G64B64_SINT)
        .value("R64G64B64_SFLOAT", Format::R64G64B64_SFLOAT)
        .value("R64G64B64A64_UINT", Format::R64G64B64A64_UINT)
        .value("R64G64B64A64_SINT", Format::R64G64B64A64_SINT)
        .value("R64G64B64A64_SFLOAT", Format::R64G64B64A64_SFLOAT)
        .value("B10G11R11_UFLOAT_PACK32", Format::B10G11R11_UFLOAT_PACK32)
        .value("E5B9G9R9_UFLOAT_PACK32", Format::E5B9G9R9_UFLOAT_PACK32)
        .value("D16_UNORM", Format::D16_UNORM)
        .value("X8_D24_UNORM_PACK32", Format::X8_D24_UNORM_PACK32)
        .value("D32_SFLOAT", Format::D32_SFLOAT)
        .value("S8_UINT", Format::S8_UINT)
        .value("D16_UNORM_S8_UINT", Format::D16_UNORM_S8_UINT)
        .value("D24_UNORM_S8_UINT", Format::D24_UNORM_S8_UINT)
        .value("D32_SFLOAT_S8_UINT", Format::D32_SFLOAT_S8_UINT)
        .value("BC1_RGB_UNORM_BLOCK", Format::BC1_RGB_UNORM_BLOCK)
        .value("BC1_RGB_SRGB_BLOCK", Format::BC1_RGB_SRGB_BLOCK)
        .value("BC1_RGBA_UNORM_BLOCK", Format::BC1_RGBA_UNORM_BLOCK)
        .value("BC1_RGBA_SRGB_BLOCK", Format::BC1_RGBA_SRGB_BLOCK)
        .value("BC2_UNORM_BLOCK", Format::BC2_UNORM_BLOCK)
        .value("BC2_SRGB_BLOCK", Format::BC2_SRGB_BLOCK)
        .value("BC3_UNORM_BLOCK", Format::BC3_UNORM_BLOCK)
        .value("BC3_SRGB_BLOCK", Format::BC3_SRGB_BLOCK)
        .value("BC4_UNORM_BLOCK", Format::BC4_UNORM_BLOCK)
        .value("BC4_SNORM_BLOCK", Format::BC4_SNORM_BLOCK)
        .value("BC5_UNORM_BLOCK", Format::BC5_UNORM_BLOCK)
        .value("BC5_SNORM_BLOCK", Format::BC5_SNORM_BLOCK)
        .value("BC6H_UFLOAT_BLOCK", Format::BC6H_UFLOAT_BLOCK)
        .value("BC6H_SFLOAT_BLOCK", Format::BC6H_SFLOAT_BLOCK)
        .value("BC7_UNORM_BLOCK", Format::BC7_UNORM_BLOCK)
        .value("BC7_SRGB_BLOCK", Format::BC7_SRGB_BLOCK)
        .value("ETC2_R8G8B8_UNORM_BLOCK", Format::ETC2_R8G8B8_UNORM_BLOCK)
        .value("ETC2_R8G8B8_SRGB_BLOCK", Format::ETC2_R8G8B8_SRGB_BLOCK)
        .value("ETC2_R8G8B8A1_UNORM_BLOCK", Format::ETC2_R8G8B8A1_UNORM_BLOCK)
        .value("ETC2_R8G8B8A1_SRGB_BLOCK", Format::ETC2_R8G8B8A1_SRGB_BLOCK)
        .value("ETC2_R8G8B8A8_UNORM_BLOCK", Format::ETC2_R8G8B8A8_UNORM_BLOCK)
        .value("ETC2_R8G8B8A8_SRGB_BLOCK", Format::ETC2_R8G8B8A8_SRGB_BLOCK)
        .value("EAC_R11_UNORM_BLOCK", Format::EAC_R11_UNORM_BLOCK)
        .value("EAC_R11_SNORM_BLOCK", Format::EAC_R11_SNORM_BLOCK)
        .value("EAC_R11G11_UNORM_BLOCK", Format::EAC_R11G11_UNORM_BLOCK)
        .value("EAC_R11G11_SNORM_BLOCK", Format::EAC_R11G11_SNORM_BLOCK)
        .value("ASTC_4x4_UNORM_BLOCK", Format::ASTC_4x4_UNORM_BLOCK)
        .value("ASTC_4x4_SRGB_BLOCK", Format::ASTC_4x4_SRGB_BLOCK)
        .value("ASTC_5x4_UNORM_BLOCK", Format::ASTC_5x4_UNORM_BLOCK)
        .value("ASTC_5x4_SRGB_BLOCK", Format::ASTC_5x4_SRGB_BLOCK)
        .value("ASTC_5x5_UNORM_BLOCK", Format::ASTC_5x5_UNORM_BLOCK)
        .value("ASTC_5x5_SRGB_BLOCK", Format::ASTC_5x5_SRGB_BLOCK)
        .value("ASTC_6x5_UNORM_BLOCK", Format::ASTC_6x5_UNORM_BLOCK)
        .value("ASTC_6x5_SRGB_BLOCK", Format::ASTC_6x5_SRGB_BLOCK)
        .value("ASTC_6x6_UNORM_BLOCK", Format::ASTC_6x6_UNORM_BLOCK)
        .value("ASTC_6x6_SRGB_BLOCK", Format::ASTC_6x6_SRGB_BLOCK)
        .value("ASTC_8x5_UNORM_BLOCK", Format::ASTC_8x5_UNORM_BLOCK)
        .value("ASTC_8x5_SRGB_BLOCK", Format::ASTC_8x5_SRGB_BLOCK)
        .value("ASTC_8x6_UNORM_BLOCK", Format::ASTC_8x6_UNORM_BLOCK)
        .value("ASTC_8x6_SRGB_BLOCK", Format::ASTC_8x6_SRGB_BLOCK)
        .value("ASTC_8x8_UNORM_BLOCK", Format::ASTC_8x8_UNORM_BLOCK)
        .value("ASTC_8x8_SRGB_BLOCK", Format::ASTC_8x8_SRGB_BLOCK)
        .value("ASTC_10x5_UNORM_BLOCK", Format::ASTC_10x5_UNORM_BLOCK)
        .value("ASTC_10x5_SRGB_BLOCK", Format::ASTC_10x5_SRGB_BLOCK)
        .value("ASTC_10x6_UNORM_BLOCK", Format::ASTC_10x6_UNORM_BLOCK)
        .value("ASTC_10x6_SRGB_BLOCK", Format::ASTC_10x6_SRGB_BLOCK)
        .value("ASTC_10x8_UNORM_BLOCK", Format::ASTC_10x8_UNORM_BLOCK)
        .value("ASTC_10x8_SRGB_BLOCK", Format::ASTC_10x8_SRGB_BLOCK)
        .value("ASTC_10x10_UNORM_BLOCK", Format::ASTC_10x10_UNORM_BLOCK)
        .value("ASTC_10x10_SRGB_BLOCK", Format::ASTC_10x10_SRGB_BLOCK)
        .value("ASTC_12x10_UNORM_BLOCK", Format::ASTC_12x10_UNORM_BLOCK)
        .value("ASTC_12x10_SRGB_BLOCK", Format::ASTC_12x10_SRGB_BLOCK)
        .value("ASTC_12x12_UNORM_BLOCK", Format::ASTC_12x12_UNORM_BLOCK)
        .value("ASTC_12x12_SRGB_BLOCK", Format::ASTC_12x12_SRGB_BLOCK)
        .value("G8B8G8R8_422_UNORM", Format::G8B8G8R8_422_UNORM)
        .value("B8G8R8G8_422_UNORM", Format::B8G8R8G8_422_UNORM)
        .value("G8_B8_R8_3PLANE_420_UNORM", Format::G8_B8_R8_3PLANE_420_UNORM)
        .value("G8_B8R8_2PLANE_420_UNORM", Format::G8_B8R8_2PLANE_420_UNORM)
        .value("G8_B8_R8_3PLANE_422_UNORM", Format::G8_B8_R8_3PLANE_422_UNORM)
        .value("G8_B8R8_2PLANE_422_UNORM", Format::G8_B8R8_2PLANE_422_UNORM)
        .value("G8_B8_R8_3PLANE_444_UNORM", Format::G8_B8_R8_3PLANE_444_UNORM)
        .value("R10X6_UNORM_PACK16", Format::R10X6_UNORM_PACK16)
        .value("R10X6G10X6_UNORM_2PACK16", Format::R10X6G10X6_UNORM_2PACK16)
        .value("R10X6G10X6B10X6A10X6_UNORM_4PACK16", Format::R10X6G10X6B10X6A10X6_UNORM_4PACK16)
        .value("G10X6B10X6G10X6R10X6_422_UNORM_4PACK16", Format::G10X6B10X6G10X6R10X6_422_UNORM_4PACK16)
        .value("B10X6G10X6R10X6G10X6_422_UNORM_4PACK16", Format::B10X6G10X6R10X6G10X6_422_UNORM_4PACK16)
        .value("G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16", Format::G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16)
        .value("G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16", Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16)
        .value("G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16", Format::G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16)
        .value("G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16", Format::G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16)
        .value("G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16", Format::G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16)
        .value("R12X4_UNORM_PACK16", Format::R12X4_UNORM_PACK16)
        .value("R12X4G12X4_UNORM_2PACK16", Format::R12X4G12X4_UNORM_2PACK16)
        .value("R12X4G12X4B12X4A12X4_UNORM_4PACK16", Format::R12X4G12X4B12X4A12X4_UNORM_4PACK16)
        .value("G12X4B12X4G12X4R12X4_422_UNORM_4PACK16", Format::G12X4B12X4G12X4R12X4_422_UNORM_4PACK16)
        .value("B12X4G12X4R12X4G12X4_422_UNORM_4PACK16", Format::B12X4G12X4R12X4G12X4_422_UNORM_4PACK16)
        .value("G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16", Format::G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16)
        .value("G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16", Format::G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16)
        .value("G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16", Format::G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16)
        .value("G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16", Format::G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16)
        .value("G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16", Format::G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16)
        .value("G16B16G16R16_422_UNORM", Format::G16B16G16R16_422_UNORM)
        .value("B16G16R16G16_422_UNORM", Format::B16G16R16G16_422_UNORM)
        .value("G16_B16_R16_3PLANE_420_UNORM", Format::G16_B16_R16_3PLANE_420_UNORM)
        .value("G16_B16R16_2PLANE_420_UNORM", Format::G16_B16R16_2PLANE_420_UNORM)
        .value("G16_B16_R16_3PLANE_422_UNORM", Format::G16_B16_R16_3PLANE_422_UNORM)
        .value("G16_B16R16_2PLANE_422_UNORM", Format::G16_B16R16_2PLANE_422_UNORM)
        .value("G16_B16_R16_3PLANE_444_UNORM", Format::G16_B16_R16_3PLANE_444_UNORM)
        .value("G8_B8R8_2PLANE_444_UNORM", Format::G8_B8R8_2PLANE_444_UNORM)
        .value("G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16", Format::G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16)
        .value("G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16", Format::G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16)
        .value("G16_B16R16_2PLANE_444_UNORM", Format::G16_B16R16_2PLANE_444_UNORM)
        .value("A4R4G4B4_UNORM_PACK16", Format::A4R4G4B4_UNORM_PACK16)
        .value("A4B4G4R4_UNORM_PACK16", Format::A4B4G4R4_UNORM_PACK16)
        .value("ASTC_4x4_SFLOAT_BLOCK", Format::ASTC_4x4_SFLOAT_BLOCK)
        .value("ASTC_5x4_SFLOAT_BLOCK", Format::ASTC_5x4_SFLOAT_BLOCK)
        .value("ASTC_5x5_SFLOAT_BLOCK", Format::ASTC_5x5_SFLOAT_BLOCK)
        .value("ASTC_6x5_SFLOAT_BLOCK", Format::ASTC_6x5_SFLOAT_BLOCK)
        .value("ASTC_6x6_SFLOAT_BLOCK", Format::ASTC_6x6_SFLOAT_BLOCK)
        .value("ASTC_8x5_SFLOAT_BLOCK", Format::ASTC_8x5_SFLOAT_BLOCK)
        .value("ASTC_8x6_SFLOAT_BLOCK", Format::ASTC_8x6_SFLOAT_BLOCK)
        .value("ASTC_8x8_SFLOAT_BLOCK", Format::ASTC_8x8_SFLOAT_BLOCK)
        .value("ASTC_10x5_SFLOAT_BLOCK", Format::ASTC_10x5_SFLOAT_BLOCK)
        .value("ASTC_10x6_SFLOAT_BLOCK", Format::ASTC_10x6_SFLOAT_BLOCK)
        .value("ASTC_10x8_SFLOAT_BLOCK", Format::ASTC_10x8_SFLOAT_BLOCK)
        .value("ASTC_10x10_SFLOAT_BLOCK", Format::ASTC_10x10_SFLOAT_BLOCK)
        .value("ASTC_12x10_SFLOAT_BLOCK", Format::ASTC_12x10_SFLOAT_BLOCK)
        .value("ASTC_12x12_SFLOAT_BLOCK", Format::ASTC_12x12_SFLOAT_BLOCK)
        .value("PVRTC1_2BPP_UNORM_BLOCK_IMG", Format::PVRTC1_2BPP_UNORM_BLOCK_IMG)
        .value("PVRTC1_4BPP_UNORM_BLOCK_IMG", Format::PVRTC1_4BPP_UNORM_BLOCK_IMG)
        .value("PVRTC2_2BPP_UNORM_BLOCK_IMG", Format::PVRTC2_2BPP_UNORM_BLOCK_IMG)
        .value("PVRTC2_4BPP_UNORM_BLOCK_IMG", Format::PVRTC2_4BPP_UNORM_BLOCK_IMG)
        .value("PVRTC1_2BPP_SRGB_BLOCK_IMG", Format::PVRTC1_2BPP_SRGB_BLOCK_IMG)
        .value("PVRTC1_4BPP_SRGB_BLOCK_IMG", Format::PVRTC1_4BPP_SRGB_BLOCK_IMG)
        .value("PVRTC2_2BPP_SRGB_BLOCK_IMG", Format::PVRTC2_2BPP_SRGB_BLOCK_IMG)
        .value("PVRTC2_4BPP_SRGB_BLOCK_IMG", Format::PVRTC2_4BPP_SRGB_BLOCK_IMG)
        .value("R16G16_SFIXED5_NV", Format::R16G16_SFIXED5_NV)
        .value("A1B5G5R5_UNORM_PACK16_KHR", Format::A1B5G5R5_UNORM_PACK16_KHR)
        .value("A8_UNORM_KHR", Format::A8_UNORM_KHR)
        .value("ASTC_4x4_SFLOAT_BLOCK_EXT", Format::ASTC_4x4_SFLOAT_BLOCK_EXT)
        .value("ASTC_5x4_SFLOAT_BLOCK_EXT", Format::ASTC_5x4_SFLOAT_BLOCK_EXT)
        .value("ASTC_5x5_SFLOAT_BLOCK_EXT", Format::ASTC_5x5_SFLOAT_BLOCK_EXT)
        .value("ASTC_6x5_SFLOAT_BLOCK_EXT", Format::ASTC_6x5_SFLOAT_BLOCK_EXT)
        .value("ASTC_6x6_SFLOAT_BLOCK_EXT", Format::ASTC_6x6_SFLOAT_BLOCK_EXT)
        .value("ASTC_8x5_SFLOAT_BLOCK_EXT", Format::ASTC_8x5_SFLOAT_BLOCK_EXT)
        .value("ASTC_8x6_SFLOAT_BLOCK_EXT", Format::ASTC_8x6_SFLOAT_BLOCK_EXT)
        .value("ASTC_8x8_SFLOAT_BLOCK_EXT", Format::ASTC_8x8_SFLOAT_BLOCK_EXT)
        .value("ASTC_10x5_SFLOAT_BLOCK_EXT", Format::ASTC_10x5_SFLOAT_BLOCK_EXT)
        .value("ASTC_10x6_SFLOAT_BLOCK_EXT", Format::ASTC_10x6_SFLOAT_BLOCK_EXT)
        .value("ASTC_10x8_SFLOAT_BLOCK_EXT", Format::ASTC_10x8_SFLOAT_BLOCK_EXT)
        .value("ASTC_10x10_SFLOAT_BLOCK_EXT", Format::ASTC_10x10_SFLOAT_BLOCK_EXT)
        .value("ASTC_12x10_SFLOAT_BLOCK_EXT", Format::ASTC_12x10_SFLOAT_BLOCK_EXT)
        .value("ASTC_12x12_SFLOAT_BLOCK_EXT", Format::ASTC_12x12_SFLOAT_BLOCK_EXT)
        .value("G8B8G8R8_422_UNORM_KHR", Format::G8B8G8R8_422_UNORM_KHR)
        .value("B8G8R8G8_422_UNORM_KHR", Format::B8G8R8G8_422_UNORM_KHR)
        .value("G8_B8_R8_3PLANE_420_UNORM_KHR", Format::G8_B8_R8_3PLANE_420_UNORM_KHR)
        .value("G8_B8R8_2PLANE_420_UNORM_KHR", Format::G8_B8R8_2PLANE_420_UNORM_KHR)
        .value("G8_B8_R8_3PLANE_422_UNORM_KHR", Format::G8_B8_R8_3PLANE_422_UNORM_KHR)
        .value("G8_B8R8_2PLANE_422_UNORM_KHR", Format::G8_B8R8_2PLANE_422_UNORM_KHR)
        .value("G8_B8_R8_3PLANE_444_UNORM_KHR", Format::G8_B8_R8_3PLANE_444_UNORM_KHR)
        .value("R10X6_UNORM_PACK16_KHR", Format::R10X6_UNORM_PACK16_KHR)
        .value("R10X6G10X6_UNORM_2PACK16_KHR", Format::R10X6G10X6_UNORM_2PACK16_KHR)
        .value("R10X6G10X6B10X6A10X6_UNORM_4PACK16_KHR", Format::R10X6G10X6B10X6A10X6_UNORM_4PACK16_KHR)
        .value("G10X6B10X6G10X6R10X6_422_UNORM_4PACK16_KHR", Format::G10X6B10X6G10X6R10X6_422_UNORM_4PACK16_KHR)
        .value("B10X6G10X6R10X6G10X6_422_UNORM_4PACK16_KHR", Format::B10X6G10X6R10X6G10X6_422_UNORM_4PACK16_KHR)
        .value("G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16_KHR", Format::G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16_KHR)
        .value("G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16_KHR", Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16_KHR)
        .value("G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16_KHR", Format::G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16_KHR)
        .value("G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16_KHR", Format::G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16_KHR)
        .value("G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16_KHR", Format::G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16_KHR)
        .value("R12X4_UNORM_PACK16_KHR", Format::R12X4_UNORM_PACK16_KHR)
        .value("R12X4G12X4_UNORM_2PACK16_KHR", Format::R12X4G12X4_UNORM_2PACK16_KHR)
        .value("R12X4G12X4B12X4A12X4_UNORM_4PACK16_KHR", Format::R12X4G12X4B12X4A12X4_UNORM_4PACK16_KHR)
        .value("G12X4B12X4G12X4R12X4_422_UNORM_4PACK16_KHR", Format::G12X4B12X4G12X4R12X4_422_UNORM_4PACK16_KHR)
        .value("B12X4G12X4R12X4G12X4_422_UNORM_4PACK16_KHR", Format::B12X4G12X4R12X4G12X4_422_UNORM_4PACK16_KHR)
        .value("G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16_KHR", Format::G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16_KHR)
        .value("G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16_KHR", Format::G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16_KHR)
        .value("G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16_KHR", Format::G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16_KHR)
        .value("G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16_KHR", Format::G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16_KHR)
        .value("G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16_KHR", Format::G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16_KHR)
        .value("G16B16G16R16_422_UNORM_KHR", Format::G16B16G16R16_422_UNORM_KHR)
        .value("B16G16R16G16_422_UNORM_KHR", Format::B16G16R16G16_422_UNORM_KHR)
        .value("G16_B16_R16_3PLANE_420_UNORM_KHR", Format::G16_B16_R16_3PLANE_420_UNORM_KHR)
        .value("G16_B16R16_2PLANE_420_UNORM_KHR", Format::G16_B16R16_2PLANE_420_UNORM_KHR)
        .value("G16_B16_R16_3PLANE_422_UNORM_KHR", Format::G16_B16_R16_3PLANE_422_UNORM_KHR)
        .value("G16_B16R16_2PLANE_422_UNORM_KHR", Format::G16_B16R16_2PLANE_422_UNORM_KHR)
        .value("G16_B16_R16_3PLANE_444_UNORM_KHR", Format::G16_B16_R16_3PLANE_444_UNORM_KHR)
        .value("G8_B8R8_2PLANE_444_UNORM_EXT", Format::G8_B8R8_2PLANE_444_UNORM_EXT)
        .value("G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16_EXT", Format::G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16_EXT)
        .value("G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16_EXT", Format::G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16_EXT)
        .value("G16_B16R16_2PLANE_444_UNORM_EXT", Format::G16_B16R16_2PLANE_444_UNORM_EXT)
        .value("A4R4G4B4_UNORM_PACK16_EXT", Format::A4R4G4B4_UNORM_PACK16_EXT)
        .value("A4B4G4R4_UNORM_PACK16_EXT", Format::A4B4G4R4_UNORM_PACK16_EXT)
        .value("R16G16_S10_5_NV", Format::R16G16_S10_5_NV)
    ;

    nb::class_<VertexAttribute>(m, "VertexAttribute")
        .def(nb::init<u32, u32, Format, u32>(), nb::arg("location"), nb::arg("binding"), nb::arg("format"), nb::arg("offset") = 0)
    ;

    nb::enum_<PrimitiveTopology>(m, "PrimitiveTopology")
        .value("POINT_LIST",                     PrimitiveTopology::PointList)
        .value("LINE_LIST",                      PrimitiveTopology::LineList)
        .value("LINE_STRIP",                     PrimitiveTopology::LineStrip)
        .value("TRIANGLE_LIST",                  PrimitiveTopology::TriangleList)
        .value("TRIANGLE_STRIP",                 PrimitiveTopology::TriangleStrip)
        .value("TRIANGLE_FAN",                   PrimitiveTopology::TriangleFan)
        .value("LINE_LIST_WITH_ADJACENCY",       PrimitiveTopology::LineListWithAdjacency)
        .value("LINE_STRIP_WITH_ADJACENCY",      PrimitiveTopology::LineStripWithAdjacency)
        .value("TRIANGLE_LIST_WITH_ADJACENCY",   PrimitiveTopology::TriangleListWithAdjacency)
        .value("TRIANGLE_STRIP_WITH_ADJACENCY",  PrimitiveTopology::TriangleStripWithAdjacency)
        .value("PATCH_LIST",                     PrimitiveTopology::PatchList)
    ;

    nb::class_<InputAssembly>(m, "InputAssembly")
        .def(nb::init<PrimitiveTopology, bool>(), nb::arg("primitive_topology") = PrimitiveTopology::TriangleList, nb::arg("primitive_restart_enable") = false);
    ;

    nb::enum_<DescriptorType>(m, "DescriptorType")
        .value("SAMPLER", DescriptorType::Sampler)
        .value("COMBINED_IMAGE_SAMPLER", DescriptorType::CombinedImageSampler)
        .value("SAMPLED_IMAGE", DescriptorType::SampledImage)
        .value("STORAGE_IMAGE", DescriptorType::StorageImage)
        .value("UNIFORM_TEXEL_BUFFER", DescriptorType::UniformTexelBuffer)
        .value("STORAGE_TEXEL_BUFFER", DescriptorType::StorageTexelBuffer)
        .value("UNIFORM_BUFFER", DescriptorType::UniformBuffer)
        .value("STORAGE_BUFFER", DescriptorType::StorageBuffer)
        .value("UNIFORM_BUFFER_DYNAMIC", DescriptorType::UniformBufferDynamic)
        .value("STORAGE_BUFFER_DYNAMIC", DescriptorType::StorageBufferDynamic)
        .value("INPUT_ATTACHMENT", DescriptorType::InputAttachment)
        .value("INLINE_UNIFORM_BLOCK", DescriptorType::InlineUniformBlock)
        .value("ACCELERATION_STRUCTURE", DescriptorType::AccelerationStructure)
        .value("SAMPLE_WEIGHT_IMAGE", DescriptorType::SampleWeightImage)
        .value("BLOCK_MATCH_IMAGE", DescriptorType::BlockMatchImage)
        .value("MUTABLE", DescriptorType::Mutable)
    ;

    nb::class_<DescriptorSetEntry>(m, "DescriptorSetEntry")
        .def(nb::init<u32, DescriptorType>(), nb::arg("count"), nb::arg("type"))
    ;

    nb::enum_<DescriptorBindingFlags>(m, "DescriptorBindingFlags", nb::is_arithmetic() , nb::is_flag())
        .value("UPDATE_AFTER_BIND",           DescriptorBindingFlags::UpdateAfterBind)
        .value("UPDATE_UNUSED_WHILE_PENDING", DescriptorBindingFlags::UpdateUnusedWhilePending)
        .value("PARTIALLY_BOUND",             DescriptorBindingFlags::PartiallyBound)
        .value("VARIABLE_DESCRIPTOR_COUNT",   DescriptorBindingFlags::VariableDescriptorCount)
    ;

    nb::class_<DescriptorSet>(m, "DescriptorSet",
        nb::intrusive_ptr<DescriptorSet>([](DescriptorSet *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<nb::ref<Context>, const std::vector<DescriptorSetEntry>&, DescriptorBindingFlags>(), nb::arg("ctx"), nb::arg("entries"), nb::arg("flags") = DescriptorBindingFlags())
    ;

    nb::class_<GraphicsPipeline>(m, "GraphicsPipeline",
        nb::intrusive_ptr<GraphicsPipeline>([](GraphicsPipeline *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<nb::ref<Context>,
                const std::vector<nb::ref<PipelineStage>>&,
                const std::vector<VertexBinding>&,
                const std::vector<VertexAttribute>&,
                InputAssembly,
                const std::vector<nb::ref<DescriptorSet>>&
            >(),
            nb::arg("ctx"),
            nb::arg("stages") = std::vector<nb::ref<PipelineStage>>(),
            nb::arg("vertex_bindings") = std::vector<VertexBinding>(),
            nb::arg("vertex_attributes") = std::vector<VertexAttribute>(),
            nb::arg("input_assembly") = InputAssembly(),
            nb::arg("descriptor_sets") = std::vector<nb::ref<DescriptorSet>>()
        )
        .def("destroy", &GraphicsPipeline::destroy)
    ;

    nb::class_<Frame>(m, "Frame")
        .def_prop_ro("command_pool", [](Frame& f) -> CommandPool {
            return CommandPool(f.frame.command_pool);
        })
        .def_prop_ro("command_buffer", [](Frame& f) -> CommandBuffer {
            return CommandBuffer(f.frame.command_buffer);
        })
    ;

    nb::class_<CommandPool>(m, "CommandPool");
    nb::class_<CommandBuffer>(m, "CommandBuffer");

    nb::enum_<gfx::SwapchainStatus>(m, "SwapchainStatus")
        .value("READY", gfx::SwapchainStatus::READY)
        .value("RESIZED", gfx::SwapchainStatus::RESIZED)
        .value("MINIMIZED", gfx::SwapchainStatus::MINIMIZED)
    ;

    m.def("process_events", &gfx::ProcessEvents, nb::arg("wait"));
    m.def("begin_commands", &BeginCommands, nb::arg("command_pool"), nb::arg("command_buffer"), nb::arg("ctx"));
    m.def("end_commands", &EndCommands, nb::arg("command_buffer"));

    // m.def("test", &test, nb::arg("callback"));
    // m.def("test2", &test2, nb::arg("callback"));

    nb::module_ mod_imgui = m.def_submodule("imgui", "ImGui bindings for XPG");
    #include "generated_imgui.inc"

    // TODO: missing likely more
    // nb::class_<ImVec2>(mod_imgui, "IVec2")
    //     .def(nb::init<>())
    //     .def(nb::init<int32_t, int32_t>())
    //     .def_rw("x", &glm::ivec2::x)
    //     .def_rw("y", &glm::ivec2::y);

    nb::class_<ImVec2>(mod_imgui, "Vec2")
        .def(nb::init<>())
        .def(nb::init<float, float>())
        .def_rw("x", &ImVec2::x)
        .def_rw("y", &ImVec2::y);
    nb::class_<ImVec4>(mod_imgui, "Vec4")
        .def(nb::init<>())
        .def(nb::init<float, float, float, float>())
        .def_rw("x", &ImVec4::x)
        .def_rw("y", &ImVec4::y)
        .def_rw("z", &ImVec4::z)
        .def_rw("w", &ImVec4::w);

    // Examples:
    // nb::class_<ImGuiStyle>(mod_imgui, "Style")
    //     .def_rw("alpha", &ImGuiStyle::Alpha)
    //     .def_rw("window_padding", &ImGuiStyle::WindowPadding);

    // nb::enum_<WindowFlags>(m2, "WindowFlags")
    //     .value("NONE", WindowFlags::NONE)
    //     .value("NO_TITLE_BAR", WindowFlags::NO_TITLE_BAR)
    //     .export_values();

    // mod_imgui.def("begin", ImGui_begin);
    // mod_imgui.def("get_style", ImGui_get_style, nb::rv_policy::reference);


    // SLANG
    // nb::module_ mod_slang = m.def_submodule("slang", "Slang bindings for XPG");
    // nb::enum_<slang::TypeReflection::Kind>(mod_slang, "TypeKind")
    //   .def("None", slang::TypeReflection::Kind::None)
    //   .def("Struct", slang::TypeReflection::Kind::Struct)
    //   .def("Array", slang::TypeReflection::Kind::Array)
    //   .def("Matrix", slang::TypeReflection::Kind::Matrix)
    //   .def("Vector", slang::TypeReflection::Kind::Vector)
    //   .def("Scalar", slang::TypeReflection::Kind::Scalar)
    //   .def("ConstantBuffer", slang::TypeReflection::Kind::ConstantBuffer)
    //   .def("Resource", slang::TypeReflection::Kind::Resource)
    //   .def("SamplerState", slang::TypeReflection::Kind::SamplerState)
    //   .def("TextureBuffer", slang::TypeReflection::Kind::TextureBuffer)
    //   .def("ShaderStorageBuffer", slang::TypeReflection::Kind::ShaderStorageBuffer)
    //   .def("ParameterBlock", slang::TypeReflection::Kind::ParameterBlock)
    //   .def("GenericTypeParameter", slang::TypeReflection::Kind::GenericTypeParameter)
    //   .def("Interface", slang::TypeReflection::Kind::Interface)
    //   .def("OutputStream", slang::TypeReflection::Kind::OutputStream)
    //   .def("Specialized", slang::TypeReflection::Kind::Specialized)
    //   .def("Feedback", slang::TypeReflection::Kind::Feedback)
    //   .def("Pointer", slang::TypeReflection::Kind::Pointer)
    //   .def("DynamicResource", slang::TypeReflection::Kind::DynamicResource)
    // ;

    // nb::enum_<slang::TypeReflection::ScalarType>(mod_slang, "ScalarType")
    //     .def("None", slang::TypeReflection::ScalarType::None)
    //     .def("Void", slang::TypeReflection::ScalarType::Void)
    //     .def("Bool", slang::TypeReflection::ScalarType::Bool)
    //     .def("Int32", slang::TypeReflection::ScalarType::Int32)
    //     .def("UInt32", slang::TypeReflection::ScalarType::UInt32)
    //     .def("Int64", slang::TypeReflection::ScalarType::Int64)
    //     .def("UInt64", slang::TypeReflection::ScalarType::UInt64)
    //     .def("Float16", slang::TypeReflection::ScalarType::Float16)
    //     .def("Float32", slang::TypeReflection::ScalarType::Float32)
    //     .def("Float64", slang::TypeReflection::ScalarType::Float64)
    //     .def("Int8", slang::TypeReflection::ScalarType::Int8)
    //     .def("UInt8", slang::TypeReflection::ScalarType::UInt8)
    //     .def("Int16", slang::TypeReflection::ScalarType::Int16)
    //     .def("UInt16", slang::TypeReflection::ScalarType::UInt16)
    // ;



    // $!
    // !$





}