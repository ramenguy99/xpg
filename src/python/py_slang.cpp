#include <vector>
#include <functional>

#include <nanobind/nanobind.h>
#include <nanobind/intrusive/counter.h>
#include <nanobind/intrusive/ref.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>

#include <slang.h>
#include <slang-com-ptr.h>

#include <xpg/defines.h>

namespace nb = nanobind;

struct Reflection_Obj: nb::intrusive_base {
    static nb::object __getstate__(const Reflection_Obj& obj) {
        return nb::none();
    }

    static void __setstate__(Reflection_Obj& obj, nb::object) {
        new (&obj) Reflection_Obj();
    }
};

struct Reflection_Scalar: Reflection_Obj {
    slang::TypeReflection::ScalarType scalar;

    static slang::TypeReflection::ScalarType __getstate__(const Reflection_Scalar& obj) {
        return obj.scalar;
    }

    static void __setstate__(Reflection_Scalar& obj, slang::TypeReflection::ScalarType scalar) {
        new (&obj) Reflection_Scalar();
        obj.scalar = scalar;
    }
};

struct Reflection_Array: Reflection_Obj {
    nb::ref<Reflection_Obj> type;
    usize count;

    static std::tuple<nb::ref<Reflection_Obj>, usize> __getstate__(const Reflection_Array& obj ) {
        return std::make_tuple(obj.type, obj.count);
    }

    static void __setstate__(Reflection_Array& obj, const std::tuple<nb::ref<Reflection_Obj>, usize> array) {
        new (&obj) Reflection_Array();

        obj.type = std::get<0>(array);
        obj.count = std::get<1>(array);
    }
};

struct Reflection_Vector: Reflection_Obj {
    slang::TypeReflection::ScalarType base;
    u32 count;

    static std::tuple<slang::TypeReflection::ScalarType, u32> __getstate__(const Reflection_Vector& obj) {
        return std::make_tuple(obj.base, obj.count);
    }

    static void __setstate__(Reflection_Vector& obj, const std::tuple<slang::TypeReflection::ScalarType, u32>& vector) {
        new (&obj) Reflection_Vector();

        obj.base = std::get<0>(vector);
        obj.count = std::get<1>(vector);
    }
};

struct Reflection_Matrix: Reflection_Obj {
    slang::TypeReflection::ScalarType base;
    u32 rows;
    u32 columns;

    static std::tuple<slang::TypeReflection::ScalarType, u32, u32> __getstate__(const Reflection_Matrix& obj) {
        return std::make_tuple(obj.base, obj.rows, obj.columns);
    }

    static void __setstate__(Reflection_Matrix& obj, const std::tuple<slang::TypeReflection::ScalarType, u32, u32>& matrix) {
        new (&obj) Reflection_Matrix();

        obj.base = std::get<0>(matrix);
        obj.rows = std::get<1>(matrix);
        obj.columns = std::get<2>(matrix);
    }
};

struct Reflection_Resource: Reflection_Obj {
    enum class Kind {
        ConstantBuffer,
        StructuredBuffer,
        Texture2D,
        AccelerationStructure,
        Sampler,
    };
    Kind kind;

    // Only for StructuredBuffer
    SlangResourceShape shape;
    SlangResourceAccess access;
    slang::BindingType binding_type = slang::BindingType::Unknown;

    // For Constant buffer, StructuredBuffer and Texture2D, the underlying type. None for the others.
    nb::ref<Reflection_Obj> type;

    static std::tuple<Kind, SlangResourceShape, SlangResourceAccess, slang::BindingType, nb::ref<Reflection_Obj>>
    __getstate__(const Reflection_Resource& obj) {
        return std::make_tuple(obj.kind, obj.shape, obj.access, obj.binding_type, obj.type);
    }

    static void __setstate__(Reflection_Resource& obj, const std::tuple<Kind, SlangResourceShape, SlangResourceAccess, slang::BindingType, nb::ref<Reflection_Obj>>& resource) {
        new (&obj) Reflection_Resource();

        obj.kind = std::get<0>(resource);
        obj.shape = std::get<1>(resource);
        obj.access = std::get<2>(resource);
        obj.binding_type = std::get<3>(resource);
        obj.type = std::get<4>(resource);
    }
};

struct Reflection_Field: Reflection_Obj {
    nb::str name;
    nb::ref<Reflection_Obj> type;

    // If type is a constant
    u32 offset;
    // u32 size;

    // If type is a resource
    u32 binding;
    u32 set;
    SlangImageFormat image_format;

    static std::tuple<nb::str, nb::ref<Reflection_Obj>, u32, u32, u32, SlangImageFormat>
    __getstate__(const Reflection_Field &obj) {
        return std::make_tuple(obj.name, obj.type, obj.offset, obj.binding, obj.set, obj.image_format);
    }

    static void __setstate__(Reflection_Field& obj, const std::tuple<nb::str, nb::ref<Reflection_Obj>, u32, u32, u32 , SlangImageFormat>& field) {
        new (&obj) Reflection_Field();

        obj.name = std::get<0>(field);
        obj.type = std::get<1>(field);
        obj.offset = std::get<2>(field);
        obj.binding = std::get<3>(field);
        obj.set = std::get<4>(field);
        obj.image_format = std::get<5>(field);
    }
};

struct Reflection_Struct: Reflection_Obj {
    std::vector<nb::ref<Reflection_Field>> fields;

    static std::vector<nb::ref<Reflection_Field>> __getstate__(const Reflection_Struct& obj) {
        return obj.fields;
    }

    static void __setstate__(Reflection_Struct& obj, const std::vector<nb::ref<Reflection_Field>>& fields) {
        new (&obj) Reflection_Struct();

        obj.fields = fields;
    }
};

struct Reflection: nb::intrusive_base {
    nb::ref<Reflection_Obj> object;

    // TODO: push constants

    static nb::ref<Reflection_Obj> __getstate__(Reflection& obj) {
        return obj.object;
    }

    static void __setstate__(Reflection& obj, nb::ref<Reflection_Obj>& object) {
        new (&obj) Reflection();

        obj.object = object;
    }
};

Slang::ComPtr<slang::IGlobalSession> g_slang_global_session;

struct SlangShader: public nb::intrusive_base {
    SlangShader() {}
    SlangShader(nb::str entry, nb::bytes c, nb::ref<Reflection> refl, nb::list dependencies): entry(std::move(entry)), code(std::move(c)), reflection(refl), dependencies(std::move(dependencies)) {}

    nb::str entry;
    nb::bytes code;
    nb::ref<Reflection> reflection;
    nb::list dependencies;

    static std::tuple<nb::bytes, nb::ref<Reflection>, nb::list> __getstate__(const SlangShader& obj) {
        return std::make_tuple(obj.code, obj.reflection, obj.dependencies);
    }

    static void __setstate__(SlangShader& obj, const std::tuple<nb::bytes, nb::ref<Reflection>, nb::list>& shader) {
        new (&obj) SlangShader();

        obj.code = std::get<0>(shader);
        obj.reflection = std::get<1>(shader);
        obj.dependencies = std::get<2>(shader);
    }
};

nb::ref<Reflection_Obj> parse_type(slang::TypeLayoutReflection* type) {
    switch(type->getKind()) {
        case slang::TypeReflection::Kind::None: {
            return new Reflection_Resource();
        } break;
        case slang::TypeReflection::Kind::Scalar: {
            std::unique_ptr<Reflection_Scalar> s = std::make_unique<Reflection_Scalar>();
            s->scalar = type->getScalarType();
            return s.release();
        } break;
        case slang::TypeReflection::Kind::Vector: {
            std::unique_ptr<Reflection_Vector> v = std::make_unique<Reflection_Vector>();
            v->base = type->getElementTypeLayout()->getScalarType();
            v->count = type->getElementCount();
            return v.release();
        } break;
        case slang::TypeReflection::Kind::Matrix: {
            std::unique_ptr<Reflection_Matrix> m = std::make_unique<Reflection_Matrix>();
            m->base = type->getElementTypeLayout()->getScalarType();
            m->rows = type->getRowCount();
            m->columns = type->getColumnCount();
            return m.release();
        } break;
        case slang::TypeReflection::Kind::Array: {
            std::unique_ptr<Reflection_Array> v = std::make_unique<Reflection_Array>();
            v->type = parse_type(type->getElementTypeLayout());
            v->count = type->getElementCount();
            return v.release();
        } break;
        case slang::TypeReflection::Kind::Struct: {
            std::unique_ptr<Reflection_Struct> s = std::make_unique<Reflection_Struct>();
            for(usize i = 0; i < type->getFieldCount(); i++) {
                slang::VariableLayoutReflection* field = type->getFieldByIndex(i);

                std::unique_ptr<Reflection_Field> f = std::make_unique<Reflection_Field>();
                f->name = nb::str(field->getName());
                f->type = parse_type(field->getTypeLayout());
                f->offset = field->getOffset();
                f->binding = field->getBindingIndex();
                f->set = field->getBindingSpace();
                f->image_format = field->getImageFormat();

                u32 sub_regspace = field->getOffset(slang::ParameterCategory::SubElementRegisterSpace);
                u32 table = field->getOffset(slang::ParameterCategory::DescriptorTableSlot);
                f->set = field->getBindingSpace() + sub_regspace;
                f->binding = table;

                s->fields.push_back(f.release());
            }
            return s.release();
        } break;
        case slang::TypeReflection::Kind::Resource: {
            std::unique_ptr<Reflection_Resource> r = std::make_unique<Reflection_Resource>();

            r->shape = (SlangResourceShape)(type->getResourceShape() & SLANG_RESOURCE_BASE_SHAPE_MASK);
            r->access = type->getResourceAccess();

            int count = type->getBindingRangeCount();
            if (count > 0) {
                r->binding_type = type->getBindingRangeType(0);
            }

            switch(r->shape) {
                case SlangResourceShape::SLANG_STRUCTURED_BUFFER: {
                    r->kind = Reflection_Resource::Kind::StructuredBuffer;

                    slang::TypeLayoutReflection* content_type = type->getElementTypeLayout();
                    r->type = parse_type(content_type);

                } break;
                case SlangResourceShape::SLANG_TEXTURE_2D: {
                    r->kind = Reflection_Resource::Kind::Texture2D;
                } break;
                case SlangResourceShape::SLANG_ACCELERATION_STRUCTURE: {
                    r->kind = Reflection_Resource::Kind::AccelerationStructure;
                } break;
                default:
                    throw std::runtime_error("Unexpected resource shape in shader reflection");
            }

            return r.release();
        } break;
        case slang::TypeReflection::Kind::SamplerState: {
            std::unique_ptr<Reflection_Resource> r = std::make_unique<Reflection_Resource>();
            r->kind = Reflection_Resource::Kind::Sampler;
            int count = type->getBindingRangeCount();
            if (count > 0) {
                r->binding_type = type->getBindingRangeType(0);
            }
            return r.release();
        } break;
        case slang::TypeReflection::Kind::ConstantBuffer: {
            std::unique_ptr<Reflection_Resource> r = std::make_unique<Reflection_Resource>();
            r->kind = Reflection_Resource::Kind::ConstantBuffer;

            slang::VariableLayoutReflection* content = type->getElementVarLayout();
            slang::TypeLayoutReflection* content_type = content->getTypeLayout();
            const char* content_type_name = content_type->getName();
            r->type = parse_type(content_type);

            int count = type->getBindingRangeCount();
            if (count > 0) {
                r->binding_type = type->getBindingRangeType(0);
            }

            return r.release();
        } break;
        case slang::TypeReflection::Kind::ParameterBlock: {
            return parse_type(type->getElementTypeLayout());
        } break;
        default:
            nb::raise("Unhandled type kind while parsing shader reflection: %d", (int)type->getKind());
    }
}

struct CompilationError: public std::runtime_error
{
    CompilationError(const std::string& error): std::runtime_error(error) {}
};

nb::ref<SlangShader> slang_compile_any(std::function<slang::IModule*(slang::ISession*, ISlangBlob**)> func, const nb::str& entry, const nb::str& target) {
    if(!g_slang_global_session) {
        SlangGlobalSessionDesc desc = {
            .enableGLSL = false, // This enables glsl compat, but increases startup time by a lot, and forces generation of .bin file on first run.
        };
        slang::createGlobalSession(&desc, g_slang_global_session.writeRef());
    }

#if 0
    slang::CompilerOptionEntry options[1] = {
        {
            .name = slang::CompilerOptionName::VulkanUseEntryPointName,
            .value = {
                .intValue0 = 1,
            },
        }
    };
#endif

    slang::TargetDesc targets[] = {
        slang::TargetDesc {
            .format = SLANG_SPIRV,
            .profile = g_slang_global_session->findProfile(target.c_str()),
#if 0
            .compilerOptionEntries = options,
            .compilerOptionEntryCount = ArrayCount(options),
#endif
        },
    };

    slang::SessionDesc session_desc = {
        .targets = targets,
        .targetCount = ArrayCount(targets),
    };

    Slang::ComPtr<slang::ISession> session;
    g_slang_global_session->createSession(session_desc, session.writeRef());

    Slang::ComPtr<slang::IBlob> diagnostics;
    Slang::ComPtr<slang::IModule> mod = Slang::ComPtr<slang::IModule>(func(session.get(), diagnostics.writeRef()));

    if(!mod) {
        throw CompilationError(diagnostics ? (char *)diagnostics->getBufferPointer() : "");
    }

    Slang::ComPtr<slang::IEntryPoint> entry_point;
    mod->findEntryPointByName(entry.c_str(), entry_point.writeRef());
    if(!entry_point) {
        throw CompilationError("Entry point not found");
    }

    nb::list dependencies;
    for(usize i = 0; i < mod->getDependencyFileCount(); i++) {
        dependencies.append(nb::str(mod->getDependencyFilePath(i)));
    }

    // slang::IComponentType* components[] = { mod };
    // slang::IComponentType* components[] = { entry_point };
    slang::IComponentType* components[] = { mod, entry_point };
    Slang::ComPtr<slang::IComponentType> program;
    SlangResult result = session->createCompositeComponentType(components, ArrayCount(components), program.writeRef());
    if(SLANG_FAILED(result)) {
        throw std::runtime_error("Composite component creation failed");
    }

    Slang::ComPtr<slang::IComponentType> linked_program;
    result = program->link(linked_program.writeRef(), diagnostics.writeRef());
    if(SLANG_FAILED(result)) {
        throw CompilationError(diagnostics ? (char*)diagnostics->getBufferPointer() : "");
    }

    nb::ref<Reflection> reflection = new Reflection();

    slang::ProgramLayout* layout = program->getLayout();
    slang::VariableLayoutReflection* var = layout->getGlobalParamsVarLayout();
    const char* name = var->getName();
    nb::ref<Reflection_Obj> type = parse_type(var->getTypeLayout());

    reflection->object = type;

    Slang::ComPtr<slang::IBlob> kernel;
    result = linked_program->getEntryPointCode(0, // entryPointIndex
                                               0, // targetIndex
                                               kernel.writeRef(), diagnostics.writeRef());
    if(SLANG_FAILED(result)) {
        throw CompilationError(diagnostics ? (char*)diagnostics->getBufferPointer() : "");
    }

    return nb::ref<SlangShader>(new SlangShader(nb::str("main"), nb::bytes(kernel->getBufferPointer(), kernel->getBufferSize()), reflection, std::move(dependencies)));
}

nb::ref<SlangShader> slang_compile_str(const nb::str& str, const nb::str& entry, const nb::str& target, const nb::str& filename) {
    return slang_compile_any([&str, &filename] (slang::ISession* session, ISlangBlob** diagnostics) {
        return session->loadModuleFromSourceString(filename.c_str(), filename.c_str(), str.c_str(), diagnostics);
    }, entry, target);
}

nb::ref<SlangShader> slang_compile(const nb::str& file, const nb::str& entry, const nb::str& target) {
    return slang_compile_any([&file] (slang::ISession* session, ISlangBlob** diagnostics) {
        return session->loadModule(file.c_str(), diagnostics);
    }, entry, target);
}

void slang_create_bindings(nb::module_& mod_slang)
{
    nb::exception<CompilationError>(mod_slang, "CompilationError");

    nb::enum_<SlangImageFormat>(mod_slang, "ImageFormat")
        .value("UNKNOWN"        , SLANG_IMAGE_FORMAT_unknown)
        .value("RGBA32F"        , SLANG_IMAGE_FORMAT_rgba32f)
        .value("RGBA16F"        , SLANG_IMAGE_FORMAT_rgba16f)
        .value("RG32F"          , SLANG_IMAGE_FORMAT_rg32f)
        .value("RG16F"          , SLANG_IMAGE_FORMAT_rg16f)
        .value("R11F_G11F_B10F" , SLANG_IMAGE_FORMAT_r11f_g11f_b10f)
        .value("R32F"           , SLANG_IMAGE_FORMAT_r32f)
        .value("R16F"           , SLANG_IMAGE_FORMAT_r16f)
        .value("RGBA16"         , SLANG_IMAGE_FORMAT_rgba16)
        .value("RGB10_A2"       , SLANG_IMAGE_FORMAT_rgb10_a2)
        .value("RGBA8"          , SLANG_IMAGE_FORMAT_rgba8)
        .value("RG16"           , SLANG_IMAGE_FORMAT_rg16)
        .value("RG8"            , SLANG_IMAGE_FORMAT_rg8)
        .value("R16"            , SLANG_IMAGE_FORMAT_r16)
        .value("R8"             , SLANG_IMAGE_FORMAT_r8)
        .value("RGBA16_SNORM"   , SLANG_IMAGE_FORMAT_rgba16_snorm)
        .value("RGBA8_SNORM"    , SLANG_IMAGE_FORMAT_rgba8_snorm)
        .value("RG16_SNORM"     , SLANG_IMAGE_FORMAT_rg16_snorm)
        .value("RG8_SNORM"      , SLANG_IMAGE_FORMAT_rg8_snorm)
        .value("R16_SNORM"      , SLANG_IMAGE_FORMAT_r16_snorm)
        .value("R8_SNORM"       , SLANG_IMAGE_FORMAT_r8_snorm)
        .value("RGBA32I"        , SLANG_IMAGE_FORMAT_rgba32i)
        .value("RGBA16I"        , SLANG_IMAGE_FORMAT_rgba16i)
        .value("RGBA8I"         , SLANG_IMAGE_FORMAT_rgba8i)
        .value("RG32I"          , SLANG_IMAGE_FORMAT_rg32i)
        .value("RG16I"          , SLANG_IMAGE_FORMAT_rg16i)
        .value("RG8I"           , SLANG_IMAGE_FORMAT_rg8i)
        .value("R32I"           , SLANG_IMAGE_FORMAT_r32i)
        .value("R16I"           , SLANG_IMAGE_FORMAT_r16i)
        .value("R8I"            , SLANG_IMAGE_FORMAT_r8i)
        .value("RGBA32UI"       , SLANG_IMAGE_FORMAT_rgba32ui)
        .value("RGBA16UI"       , SLANG_IMAGE_FORMAT_rgba16ui)
        .value("RGB10_A2UI"     , SLANG_IMAGE_FORMAT_rgb10_a2ui)
        .value("RGBA8UI"        , SLANG_IMAGE_FORMAT_rgba8ui)
        .value("RG32UI"         , SLANG_IMAGE_FORMAT_rg32ui)
        .value("RG16UI"         , SLANG_IMAGE_FORMAT_rg16ui)
        .value("RG8UI"          , SLANG_IMAGE_FORMAT_rg8ui)
        .value("R32UI"          , SLANG_IMAGE_FORMAT_r32ui)
        .value("R16UI"          , SLANG_IMAGE_FORMAT_r16ui)
        .value("R8UI"           , SLANG_IMAGE_FORMAT_r8ui)
        .value("R64UI"          , SLANG_IMAGE_FORMAT_r64ui)
        .value("R64I"           , SLANG_IMAGE_FORMAT_r64i)
        .value("BGRA8"          , SLANG_IMAGE_FORMAT_bgra8)
    ;

    nb::enum_<slang::BindingType>(mod_slang, "BindingType")
        .value("UNKNOWN",                           slang::BindingType::Unknown)
        .value("SAMPLER",                           slang::BindingType::Sampler)
        .value("TEXTURE",                           slang::BindingType::Texture)
        .value("CONSTANT_BUFFER",                   slang::BindingType::ConstantBuffer)
        .value("PARAMETER_BLOCK",                   slang::BindingType::ParameterBlock)
        .value("TYPED_BUFFER",                      slang::BindingType::TypedBuffer)
        .value("RAW_BUFFER",                        slang::BindingType::RawBuffer)
        .value("COMBINED_TEXTURE_SAMPLER",          slang::BindingType::CombinedTextureSampler)
        .value("INPUT_RENDER_TARGET",               slang::BindingType::InputRenderTarget)
        .value("INLINE_UNIFORM_DATA",               slang::BindingType::InlineUniformData)
        .value("RAYTRACING_ACCELERATION_STRUCTURE", slang::BindingType::RayTracingAccelerationStructure)
        .value("VARYING_INPUT",                     slang::BindingType::VaryingInput)
        .value("VARYING_OUTPUT",                    slang::BindingType::VaryingOutput)
        .value("EXISTENTIAL_VALUE",                 slang::BindingType::ExistentialValue)
        .value("PUSH_CONSTANT",                     slang::BindingType::PushConstant)
        .value("MUTABLE_TEXTURE",                   slang::BindingType::MutableTexture)
        .value("MUTABLE_TYPED_BUFFER",              slang::BindingType::MutableTypedBuffer)
        .value("MUTABLE_RAW_BUFFER",                slang::BindingType::MutableRawBuffer)
    ;

    nb::enum_<Reflection_Resource::Kind>(mod_slang, "ResourceKind")
        .value("CONSTANT_BUFFER", Reflection_Resource::Kind::ConstantBuffer)
        .value("STRUCTURED_BUFFER", Reflection_Resource::Kind::StructuredBuffer)
        .value("TEXTURE_2D", Reflection_Resource::Kind::Texture2D)
        .value("ACCELERATION_STRUCTURE", Reflection_Resource::Kind::AccelerationStructure)
        .value("SAMPLER", Reflection_Resource::Kind::Sampler)
    ;

    nb::class_<Reflection_Obj>(mod_slang, "Type",
        nb::intrusive_ptr<Reflection_Obj>([](Reflection_Obj *o, PyObject *po) noexcept { o->set_self_py(po); }))
       .def("__getstate__", Reflection_Obj::__getstate__)
       .def("__setstate__", Reflection_Obj::__setstate__)
    ;

    nb::class_<Reflection_Scalar, Reflection_Obj>(mod_slang, "Scalar")
       .def_ro("base", &Reflection_Scalar::scalar)
       .def("__getstate__", Reflection_Scalar::__getstate__)
       .def("__setstate__", Reflection_Scalar::__setstate__)
    ;

    nb::class_<Reflection_Array, Reflection_Obj>(mod_slang, "Array")
       .def_ro("type", &Reflection_Array::type)
       .def_ro("count", &Reflection_Array::count)
       .def("__getstate__", Reflection_Array::__getstate__)
       .def("__setstate__", Reflection_Array::__setstate__)
    ;

    nb::class_<Reflection_Vector, Reflection_Obj>(mod_slang, "Vector")
       .def_ro("base", &Reflection_Vector::base)
       .def_ro("count", &Reflection_Vector::count)
       .def("__getstate__", Reflection_Vector::__getstate__)
       .def("__setstate__", Reflection_Vector::__setstate__)
    ;

    nb::class_<Reflection_Matrix, Reflection_Obj>(mod_slang, "Matrix")
       .def_ro("base", &Reflection_Matrix::base)
       .def_ro("rows", &Reflection_Matrix::rows)
       .def_ro("columns", &Reflection_Matrix::columns)
       .def("__getstate__", Reflection_Matrix::__getstate__)
       .def("__setstate__", Reflection_Matrix::__setstate__)
    ;

    nb::enum_<SlangResourceAccess>(mod_slang, "ResourceAccess")
        .value("SLANG_RESOURCE_ACCESS_NONE", SLANG_RESOURCE_ACCESS_NONE)
        .value("SLANG_RESOURCE_ACCESS_READ", SLANG_RESOURCE_ACCESS_READ)
        .value("SLANG_RESOURCE_ACCESS_READ_WRITE", SLANG_RESOURCE_ACCESS_READ_WRITE)
        .value("SLANG_RESOURCE_ACCESS_RASTER_ORDERED", SLANG_RESOURCE_ACCESS_RASTER_ORDERED)
        .value("SLANG_RESOURCE_ACCESS_APPEND", SLANG_RESOURCE_ACCESS_APPEND)
        .value("SLANG_RESOURCE_ACCESS_CONSUME", SLANG_RESOURCE_ACCESS_CONSUME)
        .value("SLANG_RESOURCE_ACCESS_WRITE", SLANG_RESOURCE_ACCESS_WRITE)
        .value("SLANG_RESOURCE_ACCESS_FEEDBACK", SLANG_RESOURCE_ACCESS_FEEDBACK)
        .value("SLANG_RESOURCE_ACCESS_UNKNOWN", SLANG_RESOURCE_ACCESS_UNKNOWN)
    ;

    nb::enum_<SlangResourceShape>(mod_slang, "ResourceShape")
        .value("NONE",                   SLANG_RESOURCE_NONE)
        .value("TEXTURE_1D",             SLANG_TEXTURE_1D)
        .value("TEXTURE_2D",             SLANG_TEXTURE_2D)
        .value("TEXTURE_3D",             SLANG_TEXTURE_3D)
        .value("TEXTURE_CUBE",           SLANG_TEXTURE_CUBE)
        .value("TEXTURE_BUFFER",         SLANG_TEXTURE_BUFFER)
        .value("STRUCTURED_BUFFER",      SLANG_STRUCTURED_BUFFER)
        .value("BYTE_ADDRESS_BUFFER",    SLANG_BYTE_ADDRESS_BUFFER)
        .value("RESOURCE_UNKNOWN",       SLANG_RESOURCE_UNKNOWN)
        .value("ACCELERATION_STRUCTURE", SLANG_ACCELERATION_STRUCTURE)
    ;

    nb::class_<Reflection_Resource, Reflection_Obj>(mod_slang, "Resource")
        .def_ro("kind", &Reflection_Resource::kind)
        .def_ro("shape", &Reflection_Resource::shape)
        .def_ro("access", &Reflection_Resource::access)
        .def_ro("type", &Reflection_Resource::type)
        .def_ro("binding_type", &Reflection_Resource::binding_type)
        .def("__getstate__", Reflection_Resource::__getstate__)
        .def("__setstate__", Reflection_Resource::__setstate__)
    ;

    nb::class_<Reflection_Field, Reflection_Obj>(mod_slang, "Field")
       .def_ro("name", &Reflection_Field::name)
       .def_ro("type", &Reflection_Field::type)
       .def_ro("offset", &Reflection_Field::offset)
       .def_ro("set", &Reflection_Field::set)
       .def_ro("binding", &Reflection_Field::binding)
       .def_ro("image_format", &Reflection_Field::image_format)
       .def("__getstate__", Reflection_Field::__getstate__)
       .def("__setstate__", Reflection_Field::__setstate__)
    ;

    nb::class_<Reflection_Struct, Reflection_Obj>(mod_slang, "Struct")
       .def_ro("fields", &Reflection_Struct::fields)
       .def("__getstate__", Reflection_Struct::__getstate__)
       .def("__setstate__", Reflection_Struct::__setstate__)
    ;

    nb::class_<Reflection>(mod_slang, "Reflection",
        nb::intrusive_ptr<Reflection>([](Reflection *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def("__getstate__", Reflection::__getstate__)
        .def("__setstate__", Reflection::__setstate__)
        .def_ro("object", &Reflection::object)
    ;

    nb::class_<SlangShader>(mod_slang, "Shader",
        nb::intrusive_ptr<SlangShader>([](SlangShader *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def_ro("entry", &SlangShader::entry)
        .def_ro("code", &SlangShader::code)
        .def_ro("reflection", &SlangShader::reflection)
        .def_ro("dependencies", &SlangShader::dependencies)
        .def("__getstate__", SlangShader::__getstate__)
        .def("__setstate__", SlangShader::__setstate__)
    ;

    mod_slang.def("compile", slang_compile, nb::arg("path"), nb::arg("entry") = "main", nb::arg("target") = "spirv_1_3");
    mod_slang.def("compile_str", slang_compile_str, nb::arg("source"), nb::arg("entry") = "main", nb::arg("target") = "spirv_1_3", nb::arg("filename") = "");

    nb::enum_<slang::TypeReflection::ScalarType>(mod_slang, "ScalarKind")
        .value("NONE", slang::TypeReflection::ScalarType::None)
        .value("VOID", slang::TypeReflection::ScalarType::Void)
        .value("BOOL", slang::TypeReflection::ScalarType::Bool)
        .value("INT32", slang::TypeReflection::ScalarType::Int32)
        .value("UINT32", slang::TypeReflection::ScalarType::UInt32)
        .value("INT64", slang::TypeReflection::ScalarType::Int64)
        .value("UINT64", slang::TypeReflection::ScalarType::UInt64)
        .value("FLOAT16", slang::TypeReflection::ScalarType::Float16)
        .value("FLOAT32", slang::TypeReflection::ScalarType::Float32)
        .value("FLOAT64", slang::TypeReflection::ScalarType::Float64)
        .value("INT8", slang::TypeReflection::ScalarType::Int8)
        .value("UINT8", slang::TypeReflection::ScalarType::UInt8)
        .value("INT16", slang::TypeReflection::ScalarType::Int16)
        .value("UINT16", slang::TypeReflection::ScalarType::UInt16)
    ;
}