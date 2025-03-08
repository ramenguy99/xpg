#include <vector>
#include <nanobind/intrusive/counter.h>
#include <nanobind/intrusive/ref.h>

#include <nanobind/nanobind.h>

#include <slang.h>
#include <slang-com-ptr.h>

#include <xpg/defines.h>

namespace nb = nanobind;

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

Slang::ComPtr<slang::IGlobalSession> g_slang_global_session;

struct SlangShader: public nb::intrusive_base {
    SlangShader(nb::bytes c): code(std::move(c)) {}
    nb::bytes code;
};

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

    // IComponentType* components[] = { module, entryPoint };
    slang::IComponentType* components[] = { entry_point };
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
    
    Slang::ComPtr<slang::IBlob> kernel;
    result = linked_program->getEntryPointCode(0, // entryPointIndex
                                               0, // targetIndex
                                               kernel.writeRef(), diagnostics.writeRef());
    if(SLANG_FAILED(result)) {
        throw std::runtime_error((char*)diagnostics->getBufferPointer());
    }
    
    
    return nb::ref<SlangShader>(new SlangShader(nb::bytes(kernel->getBufferPointer(), kernel->getBufferSize())));
}

void slang_create_bindings(nb::module_& mod_slang)
{
    // SLANG

    nb::class_<SlangShader>(mod_slang, "Shader",
        nb::intrusive_ptr<SlangShader>([](SlangShader *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def_prop_ro("code", [](SlangShader& s) { return s.code; });
    ;

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

}