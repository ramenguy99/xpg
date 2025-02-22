#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/function.h>
#include <nanobind/intrusive/counter.h>
#include <nanobind/intrusive/ref.h>
#include <nanobind/intrusive/counter.inl>

#if 0
#include <slang.h>
#include <slang-com-ptr.h>
#endif

#include <xpg/gfx.h>
#include <xpg/gui.h>


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
            throw std::runtime_error("yo");
        }
    }

    void set_callbacks(std::function<void()> draw)
    {
        gfx::SetWindowCallbacks(&window, {
            .draw = [draw = move(draw)]() { draw(); },
        });
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
        { Py_tp_traverse, (void *) Window::tp_traverse },
        { Py_tp_clear, (void *) Window::tp_clear },
        { 0, nullptr }
    };
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