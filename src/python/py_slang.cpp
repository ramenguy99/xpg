#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/intrusive/counter.h>
#include <nanobind/intrusive/ref.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <slang.h>
#include <slang-com-ptr.h>

#include <xpg/defines.h>

namespace nb = nanobind;

struct Reflection_Obj: nb::intrusive_base {
};

struct Reflection_Scalar: Reflection_Obj {
    slang::TypeReflection::ScalarType scalar;
};

struct Reflection_Array: Reflection_Obj {
    nb::ref<Reflection_Obj> type;
    usize count;
};

struct Reflection_Vector: Reflection_Obj {
    slang::TypeReflection::ScalarType base;
    u32 count;
};

struct Reflection_Matrix: Reflection_Obj {
    slang::TypeReflection::ScalarType base;
    u32 rows;
    u32 columns;
};

struct Reflection_Field: Reflection_Obj {
    nb::str name;
    nb::ref<Reflection_Obj> type;
    u32 offset;
    // u32 size;
};

struct Reflection_Struct: Reflection_Obj {
    std::vector<nb::ref<Reflection_Field>> fields;
};

struct Reflection_Resource: nb::intrusive_base {
    enum class Kind {
        ConstantBuffer,
        StructuredBuffer,
    };
    Kind kind;

    // Only for StructuredBuffer
    SlangResourceShape shape;
    SlangResourceAccess access;

    nb::str name;
    u32 set;
    u32 binding;

    nb::ref<Reflection_Obj> type;
};

struct Reflection: nb::intrusive_base {
    std::vector<nb::ref<Reflection_Resource>> resources;

    // TODO: push constants
};

Slang::ComPtr<slang::IGlobalSession> g_slang_global_session;

struct SlangShader: public nb::intrusive_base {
    SlangShader(nb::bytes c, nb::ref<Reflection> refl): code(std::move(c)), reflection(refl) {}

    nb::bytes code;
    nb::ref<Reflection> reflection;
};

nb::ref<Reflection_Obj> parse_type(slang::TypeLayoutReflection* type) {
    switch(type->getKind()) {
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

                s->fields.push_back(f.release());
            }
            return s.release();
        } break;
        default:
            throw std::runtime_error("Unhandled type kind while parsing shader reflection");
    }
}

nb::ref<SlangShader> slang_compile(const nb::str& file, const::nb::str& entry) {
    if(!g_slang_global_session) {
        SlangGlobalSessionDesc desc = {
            .enableGLSL = true,
        };
        slang::createGlobalSession(&desc, g_slang_global_session.writeRef());
    }

    slang::TargetDesc targets[] = {
        slang::TargetDesc {
            .format = SLANG_SPIRV,
            .profile = g_slang_global_session->findProfile("spirv_1_6"),
        },
    };

    slang::SessionDesc session_desc = {
        .targets = targets,
        .targetCount = ArrayCount(targets),
    };

    Slang::ComPtr<slang::ISession> session;
    g_slang_global_session->createSession(session_desc, session.writeRef());

    Slang::ComPtr<slang::IBlob> diagnostics;
    Slang::ComPtr<slang::IModule> mod = Slang::ComPtr<slang::IModule>(session->loadModule(file.c_str(), diagnostics.writeRef()));

    if(!mod) {
        throw std::runtime_error((char *)diagnostics->getBufferPointer());
    }

    Slang::ComPtr<slang::IEntryPoint> entry_point;
    mod->findEntryPointByName(entry.c_str(), entry_point.writeRef());
    if(!entry_point) {
        throw std::runtime_error("Entry point not found");
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
        throw std::runtime_error((char*)diagnostics->getBufferPointer());
    }

    nb::ref<Reflection> reflection = new Reflection();

    slang::ProgramLayout* layout = program->getLayout();
    slang::VariableLayoutReflection* refl = layout->getGlobalParamsVarLayout();
    slang::TypeLayoutReflection* type_layout = refl->getTypeLayout();
    switch(type_layout->getKind()) {
        case slang::TypeReflection::Kind::Struct: {
            usize count = type_layout->getFieldCount();

            for (usize i = 0; i < count; i++) {
                nb::ref<Reflection_Resource> resource = new Reflection_Resource();

                slang::VariableLayoutReflection* var = type_layout->getFieldByIndex(i);
                slang::ParameterCategory cat = var->getCategory();
                if(cat != slang::ParameterCategory::DescriptorTableSlot) {
                    // TODO: remove
                    continue;
                    throw std::runtime_error("Unexpected parameter category in shader reflection");
                }

                resource->name = nb::str(var->getName());
                resource->binding = var->getBindingIndex();
                resource->set = var->getBindingSpace();

                slang::TypeLayoutReflection* type = var->getTypeLayout();
                switch(type->getKind()) {
                    case slang::TypeReflection::Kind::ConstantBuffer: {
                        resource->kind = Reflection_Resource::Kind::ConstantBuffer;

                        slang::VariableLayoutReflection* content = type->getElementVarLayout();
                        slang::TypeLayoutReflection* content_type = content->getTypeLayout();
                        const char* content_type_name = content_type->getName();

                        resource->type = parse_type(content_type);
                    } break;
                    case slang::TypeReflection::Kind::Resource: {
                        slang::VariableLayoutReflection* content = type->getElementVarLayout();
                        slang::TypeLayoutReflection* content_type = content->getTypeLayout();
                        const char* content_type_name = content_type->getName();

                        resource->shape = (SlangResourceShape)(type->getResourceShape() & SLANG_RESOURCE_BASE_SHAPE_MASK);
                        resource->access = type->getResourceAccess();

                        switch(resource->shape) {
                            case SlangResourceShape::SLANG_STRUCTURED_BUFFER: {
                                resource->kind = Reflection_Resource::Kind::StructuredBuffer;
                            } break;
                            default:
                                throw std::runtime_error("Unexpected resource shape in shader reflection");
                        }
                    } break;
                    default:
                        throw std::runtime_error("Unexpected variable type in shader reflection");
                }

                reflection->resources.push_back(resource);
            }
        } break;

        // not hit yet
        case slang::TypeReflection::Kind::ConstantBuffer:
        case slang::TypeReflection::Kind::ParameterBlock:

        // should never be hit
        default:
            throw std::runtime_error("Unexpected type layout kind in shader reflection");
            break;
    }
    
    Slang::ComPtr<slang::IBlob> kernel;
    result = linked_program->getEntryPointCode(0, // entryPointIndex
                                               0, // targetIndex
                                               kernel.writeRef(), diagnostics.writeRef());
    if(SLANG_FAILED(result)) {
        throw std::runtime_error((char*)diagnostics->getBufferPointer());
    }

    return nb::ref<SlangShader>(new SlangShader(nb::bytes(kernel->getBufferPointer(), kernel->getBufferSize()), reflection));
}

void slang_create_bindings(nb::module_& mod_slang)
{
    nb::class_<Reflection_Obj>(mod_slang, "Type",
        nb::intrusive_ptr<Reflection_Obj>([](Reflection_Obj *o, PyObject *po) noexcept { o->set_self_py(po); }))
    ;

    nb::class_<Reflection_Scalar, Reflection_Obj>(mod_slang, "Scalar")
       .def_ro("base", &Reflection_Scalar::scalar) 
    ;

    nb::class_<Reflection_Array, Reflection_Obj>(mod_slang, "Array")
       .def_ro("type", &Reflection_Array::type) 
       .def_ro("count", &Reflection_Array::count) 
    ;

    nb::class_<Reflection_Vector, Reflection_Obj>(mod_slang, "Vector")
       .def_ro("base", &Reflection_Vector::base) 
       .def_ro("count", &Reflection_Vector::count) 
    ;

    nb::class_<Reflection_Matrix, Reflection_Obj>(mod_slang, "Matrix")
       .def_ro("base", &Reflection_Matrix::base) 
       .def_ro("rows", &Reflection_Matrix::rows) 
       .def_ro("columns", &Reflection_Matrix::columns) 
    ;

    nb::class_<Reflection_Field, Reflection_Obj>(mod_slang, "Field")
       .def_ro("name", &Reflection_Field::name) 
       .def_ro("type", &Reflection_Field::type) 
       .def_ro("offset", &Reflection_Field::offset) 
    ;

    nb::class_<Reflection_Struct, Reflection_Obj>(mod_slang, "Struct")
       .def_ro("fields", &Reflection_Struct::fields) 
    ;

    nb::enum_<Reflection_Resource::Kind>(mod_slang, "ResourceKind")
        .value("CONSTANT_BUFFER", Reflection_Resource::Kind::ConstantBuffer)
        .value("STRUCTURED_BUFFER", Reflection_Resource::Kind::StructuredBuffer)
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

    nb::class_<Reflection_Resource>(mod_slang, "Resource",
        nb::intrusive_ptr<Reflection_Resource>([](Reflection_Resource *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def_ro("name", &Reflection_Resource::name)
        .def_ro("kind", &Reflection_Resource::kind)
        .def_ro("shape", &Reflection_Resource::shape)
        .def_ro("access", &Reflection_Resource::access)
        .def_ro("set", &Reflection_Resource::set)
        .def_ro("binding", &Reflection_Resource::binding)
        .def_ro("type", &Reflection_Resource::type)
    ;

    nb::class_<Reflection>(mod_slang, "Reflection",
        nb::intrusive_ptr<Reflection>([](Reflection *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def_ro("resources", &Reflection::resources)
    ;

    nb::class_<SlangShader>(mod_slang, "Shader",
        nb::intrusive_ptr<SlangShader>([](SlangShader *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def_prop_ro("code", [](SlangShader& s) { return s.code; })
        .def_ro("reflection", &SlangShader::reflection)
    ;

    mod_slang.def("compile", slang_compile);

    nb::enum_<slang::TypeReflection::ScalarType>(mod_slang, "ScalarKind")
        .value("None", slang::TypeReflection::ScalarType::None)
        .value("Void", slang::TypeReflection::ScalarType::Void)
        .value("Bool", slang::TypeReflection::ScalarType::Bool)
        .value("Int32", slang::TypeReflection::ScalarType::Int32)
        .value("UInt32", slang::TypeReflection::ScalarType::UInt32)
        .value("Int64", slang::TypeReflection::ScalarType::Int64)
        .value("UInt64", slang::TypeReflection::ScalarType::UInt64)
        .value("Float16", slang::TypeReflection::ScalarType::Float16)
        .value("Float32", slang::TypeReflection::ScalarType::Float32)
        .value("Float64", slang::TypeReflection::ScalarType::Float64)
        .value("Int8", slang::TypeReflection::ScalarType::Int8)
        .value("UInt8", slang::TypeReflection::ScalarType::UInt8)
        .value("Int16", slang::TypeReflection::ScalarType::Int16)
        .value("UInt16", slang::TypeReflection::ScalarType::UInt16)
    ;
}