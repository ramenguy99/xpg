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
    Buffer(nb::ref<Context> ctx, usize size, VkBufferUsageFlags usage_flags, gfx::AllocPresets::Type alloc_type)
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

    Buffer(nb::ref<Context> ctx, nb::bytes data, VkBufferUsageFlags usage_flags, gfx::AllocPresets::Type alloc_type)
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
        gfx::CreateDescriptorSet(&set, ctx->vk, {
            .entries = ArrayView((gfx::DescriptorSetEntryDesc*)entries.data(), entries.size()),
            .flags = (VkDescriptorBindingFlags)flags,
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

struct GraphicsPipeline: nb::intrusive_base {
    GraphicsPipeline(nb::ref<Context> ctx,
        const std::vector<nb::ref<PipelineStage>>& stages,
        const std::vector<VertexBinding>& vertex_bindings,
        const std::vector<VertexAttribute>& vertex_attributes,
        InputAssembly input_assembly,
        const std::vector<nb::ref<DescriptorSet>>& descriptor_sets,
        const std::vector<Attachment>& attachments
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
            .attachments = ArrayView((gfx::AttachmentDesc*)attachments.data(), attachments.size()),
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
        .def_prop_ro("swapchain_format", [](Window& w) -> VkFormat { return w.window.swapchain_format; });
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

    nb::class_<Buffer>(m, "Buffer",
        nb::intrusive_ptr<Buffer>([](Buffer *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<nb::ref<Context>, size_t, VkBufferUsageFlags, gfx::AllocPresets::Type>(), nb::arg("ctx"), nb::arg("size"), nb::arg("usage_flags"), nb::arg("alloc_type"))
        .def(nb::init<nb::ref<Context>, nb::bytes, VkBufferUsageFlags, gfx::AllocPresets::Type>(), nb::arg("ctx"), nb::arg("data"), nb::arg("usage_flags"), nb::arg("alloc_type"))
        .def("destroy", &Buffer::destroy)
    ;

    nb::class_<Shader>(m, "Shader",
        nb::intrusive_ptr<Shader>([](Shader *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<nb::ref<Context>, nb::bytes>(), nb::arg("ctx"), nb::arg("code"))
        .def("destroy", &Shader::destroy)
    ;

    nb::enum_<VkShaderStageFlagBits>(m, "Stage")
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
            );
    ;

    nb::class_<GraphicsPipeline>(m, "GraphicsPipeline",
        nb::intrusive_ptr<GraphicsPipeline>([](GraphicsPipeline *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def(nb::init<nb::ref<Context>,
                const std::vector<nb::ref<PipelineStage>>&,
                const std::vector<VertexBinding>&,
                const std::vector<VertexAttribute>&,
                InputAssembly,
                const std::vector<nb::ref<DescriptorSet>>&,
                const std::vector<Attachment>&
            >(),
            nb::arg("ctx"),
            nb::arg("stages") = std::vector<nb::ref<PipelineStage>>(),
            nb::arg("vertex_bindings") = std::vector<VertexBinding>(),
            nb::arg("vertex_attributes") = std::vector<VertexAttribute>(),
            nb::arg("input_assembly") = InputAssembly(),
            nb::arg("descriptor_sets") = std::vector<nb::ref<DescriptorSet>>(),
            nb::arg("attachments") = std::vector<Attachment>()
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