#include <stdio.h>

#include <spirv_cross.hpp>
#include <spirv_common.hpp>

#include <xpg/platform.h>
#include <xpg/array.h>
#include <xpg/log.h>

#include <glslang/build_info.h>
#include <glslang/Include/Types.h>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <SPIRV/GlslangToSpv.h>

using namespace xpg;

void print_indent(u32 indent) {
    for (u32 i = 0; i < indent; i++) {
        printf(" ");
    }
}
void print_type(const spirv_cross::Compiler& comp, const spirv_cross::SPIRType& type, u32 indent) {
    // Array
    if (!type.array.empty())
    {
        // Get array stride, e.g. float4 foo[]; Will have array stride of 16 bytes.
        // size_t array_stride = comp.type_struct_member_array_stride(type, i);
        printf("[%u]\n", type.array[0]);
    }

    if (type.basetype == spirv_cross::SPIRType::BaseType::Struct) {
        usize size = comp.get_declared_struct_size(type);
        // printf("STRUCT: %zu bytes\n", size);

        unsigned member_count = type.member_types.size();
        for (unsigned i = 0; i < member_count; i++)
        {
            const spirv_cross::SPIRType& member_type = comp.get_type(type.member_types[i]);

            size_t member_size = comp.get_declared_struct_member_size(type, i);
            size_t member_offset = comp.type_struct_member_offset(type, i);
            const std::string &name = comp.get_member_name(type.self, i);

            print_indent(indent + 4);
            printf("(%3zu, %3zu) %-10s", member_offset, member_size, name.c_str());
            print_type(comp, member_type, indent + 4);
        }
    } else {
        const char* names[] = {
		    "Unknown",
		    "Void",
		    "Boolean",
		    "SByte",
		    "UByte",
		    "Short",
		    "UShort",
		    "Int",
		    "UInt",
		    "Int64",
		    "UInt64",
		    "AtomicCounter",
		    "Half",
		    "Float",
		    "Double",
		    "Struct",
		    "Image",
		    "SampledImage",
		    "Sampler",
		    "AccelerationStructure",
		    "RayQuery",
		    "ControlPointArray",
		    "Interpolant",
		    "Char",
		    "MeshGridProperties",
        };

        // print_indent(indent + 4);
        printf("%s", type.basetype < ArrayCount(names) ? names[type.basetype] : "<?>");

        // Matrix
        if (type.columns > 1)
        {
            // Get bytes stride between columns (if column major), for float4x4 -> 16 bytes.
            // size_t matrix_stride = comp.type_struct_member_matrix_stride(type, i);
            printf(" mat%ux%u",  type.columns, type.vecsize);
        } else if(type.vecsize > 1) {
            printf(" vec%u", type.vecsize);
        }

        printf("\n");
    }
}

int main(int argc, const char** argv)
{
    if (true) {
        if (argc < 2) {
            printf("USAGE: %s FILE", argv[0]);
            exit(1);
        }

        int spirvGeneratorVersion = glslang::GetSpirvGeneratorVersion();
        printf("%d:%d.%d.%d%s\n", spirvGeneratorVersion, GLSLANG_VERSION_MAJOR, GLSLANG_VERSION_MINOR,
                GLSLANG_VERSION_PATCH, GLSLANG_VERSION_FLAVOR);

        const char* filename = argv[1];
        Array<u8> src;
        platform::Result res = platform::ReadEntireFile(filename, &src);
        if (res != platform::Result::Success) {
            logging::error("spirvrefl", "Failed to open file: %d", res);
            exit(1);
        }

        glslang::TShader shader(EShLangVertex);
        const char* string = (const char*)src.data;
        int length = (int)src.length;
        shader.setStringsWithLengthsAndNames(&string, &length, &filename, 1);

        int ClientInputSemanticsVersion = 100;
        auto Client = glslang::EShClientVulkan;
        auto ClientVersion = glslang::EShTargetVulkan_1_1;
        auto TargetLanguage = glslang::EShTargetSpv;
        auto TargetVersion = glslang::EShTargetSpv_1_3;

        // shader->setDebugInfo(true);
        shader.setEnvInput(glslang::EShSourceGlsl, EShLangVertex, Client, ClientInputSemanticsVersion);
        shader.setEnvClient(Client, ClientVersion);
        shader.setEnvTarget(TargetLanguage, TargetVersion);
        // Useful to add defines / preamble
        // shader->setPreamble(PreambleString.c_str());
        // shader->addProcesses(Processes);

        // Useful to shift bindings
        // shader->setShiftBinding(res, baseBinding[res][compUnit.stage]);
        // shader->setShiftBindingForSet(res, baseBinding[res][compUnit.stage]);

        // TODO: bunch of other options exist, see: https://github.com/KhronosGroup/glslang/blob/main/StandAlone/StandAlone.cpp

        // Parse shader
        // TODO: includer for directories
        glslang::TShader::ForbidIncluder includer;
        if(!shader.parse(GetResources(), 460, false, EShMsgDefault, includer)) {
            printf("INFO: %s\n", shader.getInfoLog());
            printf("INFO DEBUG: %s\n", shader.getInfoDebugLog());
            printf("Compilation error!\n");
            exit(1);
        }

        glslang::TProgram program;
        program.addShader(&shader);

        if(!program.link(EShMsgDefault)) {
            printf("Linking error!\n");
            exit(1);
        }

        // Assignment of resources to bindings / sets, can be customized with callbacks
        if(!program.mapIO()) {
            printf("Map IO error!\n");
            exit(1);
        }

        // Print logs
        printf("INFO: %s\n", program.getInfoLog());
        printf("INFO DEBUG: %s\n", program.getInfoDebugLog());

        // Do reflection if needed. 
        printf("REFLECTION\n");
        program.buildReflection(~0); // Dump everything
        // program.dumpReflection();

        
        for (int i = 0; i <  program.getNumUniformBlocks(); i++) {
            const glslang::TObjectReflection& ublock = program.getUniformBlock(i);
            ublock.dump();

            const glslang::TType* type = ublock.getType();
            const glslang::TQualifier qual = type->getQualifier();
            printf("SET: %d\n", qual.layoutSet);
        }

        // Output SPIR-V
        glslang::TIntermediate* intermediate = program.getIntermediate(EShLangVertex);

        spv::SpvBuildLogger logger;
        glslang::SpvOptions spv_options;
        spv_options.generateDebugInfo = false;
        spv_options.emitNonSemanticShaderDebugInfo = false;
        spv_options.emitNonSemanticShaderDebugSource = false;
        spv_options.stripDebugInfo = false;

        spv_options.disableOptimizer = false;
        spv_options.optimizeSize = false;
        spv_options.disassemble = false;
        spv_options.validate = false;
        spv_options.compileOnly = false;

        std::vector<u32> spirv;
        spv::SpvBuildLogger spv_logger;
        glslang::GlslangToSpv(*intermediate, spirv, &spv_logger, &spv_options);
        printf("Got %zu opcodes\n", spirv.size());

        // Array<u8> data;
        // platform::Result res = platform::ReadEntireFile(argv[1], &data);
        // if (res != platform::Result::Success) {
        //     logging::error("spirvrefl", "Failed to open file: %d", res);
        //     exit(1);
        // }

        // ArrayView<u32> spirv = ArrayView(data).as_view<u32>();
        // logging::info("spirvref", "Read %zu opcodes", spirv.length);

        // spirv_cross::Compiler comp(spirv.data, spirv.length);

        spirv_cross::Compiler comp(spirv);
        spirv_cross::ShaderResources resources = comp.get_shader_resources();
        for (auto& buf: resources.uniform_buffers) {
            u32 set = comp.get_decoration(buf.id, spv::DecorationDescriptorSet);
            u32 binding = comp.get_decoration(buf.id, spv::DecorationBinding);

            printf("(%u, %u) UNIFORM: %s %s\n", set, binding, buf.name.c_str(), comp.get_name(buf.id).c_str());
            const spirv_cross::SPIRType& base_type = comp.get_type(buf.base_type_id);
            print_type(comp, base_type, 4);

        }
    }
}