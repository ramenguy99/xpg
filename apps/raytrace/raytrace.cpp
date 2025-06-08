#include <xpg/gui.h>
#include <xpg/log.h>
#include <xpg/platform.h>

using glm::vec2;
using glm::ivec2;
using glm::vec3;
using glm::vec4;
using glm::mat4;

using namespace xpg;

struct MaterialParameter {
    enum Kind: u32 {
        None,
        Texture,
        Vec2,
        Vec3,
        Vec4,
    };
    Kind kind;
    union {
        u32 texture;
        vec2 v2;
        vec3 v3;
        vec4 v4;
    };
};

struct Material {
    MaterialParameter base_color;
    MaterialParameter normal;
    MaterialParameter specular;
    MaterialParameter emissive;
};

struct Mesh {
    ArrayView<vec3> positions;
    ArrayView<vec3> normals;
    ArrayView<vec3> tangents;
    ArrayView<vec2> uvs;
    ArrayView<u32> indices;

    mat4 transform;
    Material material;
};

enum class Format: u32 {
    RGBA8,
    SRGBA8,
    RGBA8_BC7,
    SRGBA8_BC7,
};

struct Image {
    u32 width;
    u32 height;
    Format format;
    ArrayView<u8> data;
};

struct Scene {
    ObjArray<Mesh> meshes;
    ObjArray<Image> images;
};


template<typename T>
ArrayView<T> consume_vec(ArrayView<u8>& buf) {
    u64 length = buf.consume<u64>();
    assert(length % sizeof(T) == 0);
    return buf.consume_view<T>(length / sizeof(T));
}

MaterialParameter consume_material_parameter(ArrayView<u8>& buf) {
    MaterialParameter p = {};
    p.kind = buf.consume<MaterialParameter::Kind>();
    switch(p.kind) {
        case MaterialParameter::Kind::None:                                   ; break;
        case MaterialParameter::Kind::Texture: p.texture = buf.consume<u32 >(); break;
        case MaterialParameter::Kind::Vec2:    p.v2      = buf.consume<vec2>(); break;
        case MaterialParameter::Kind::Vec3:    p.v3      = buf.consume<vec3>(); break;
        case MaterialParameter::Kind::Vec4:    p.v4      = buf.consume<vec4>(); break;
    }
    return p;
}

Scene parse(ArrayView<u8> buf) {
    Scene s = {};

    u64 num_meshes = buf.consume<u64>();
    for(usize i = 0; i < num_meshes; i++) {
        Mesh m = {};
        m.positions = consume_vec<vec3>(buf);
        m.normals = consume_vec<vec3>(buf);
        m.tangents = consume_vec<vec3>(buf);
        m.uvs = consume_vec<vec2>(buf);
        m.indices = consume_vec<u32>(buf);
        m.transform = buf.consume<mat4>();

        m.material.base_color = consume_material_parameter(buf);
        m.material.normal = consume_material_parameter(buf);
        m.material.specular = consume_material_parameter(buf);
        m.material.emissive = consume_material_parameter(buf);

        s.meshes.add(m);
    }

    usize total_uncompressed = 0;
    usize total_bc7 = 0;

    u64 num_images = buf.consume<u64>();
    for(usize i = 0; i < num_images; i++) {
        Image img = {};
        img.width = buf.consume<u32>();
        img.height = buf.consume<u32>();
        img.format = buf.consume<Format>();
        img.data = consume_vec<u8>(buf);
        s.images.add(img);

        switch(img.format) {
            case Format::RGBA8:
            case Format::SRGBA8:
                total_uncompressed += 1; break;
            case Format::RGBA8_BC7:
            case Format::SRGBA8_BC7:
                total_bc7 += 1; break;
        }
    }
    logging::info("raytrace", "Images uncompressed: %zu", total_uncompressed);
    logging::info("raytrace", "Images bc7: %zu", total_bc7);

    assert(buf.length == 0);
    return s;
}

int main(int argc, char** argv) {
    const char* scene_path = "res/bistro.bin";
    Array<u8> data;
    platform::Result res = platform::ReadEntireFile(scene_path, &data);
    if(res != platform::Result::Success) {
        logging::error("raytrace", "scene file not found: %s", scene_path);
        exit(100);
    }

    Scene scene = parse(data);
    logging::info("raytrace", "Meshes: %zu", scene.meshes.length);
    logging::info("raytrace", "Images: %zu", scene.images.length);

    gfx::Result result;
    result = gfx::Init();
    if (result != gfx::Result::SUCCESS) {
        logging::error("raytrace", "Failed to initialize platform\n");
    }

    gfx::Context vk = {};
    result = gfx::CreateContext(&vk, {
        .minimum_api_version = (u32)VK_API_VERSION_1_1,
        .required_features = 
            gfx::DeviceFeatures::DYNAMIC_RENDERING |
            gfx::DeviceFeatures::DESCRIPTOR_INDEXING |
            gfx::DeviceFeatures::SYNCHRONIZATION_2 | 
            gfx::DeviceFeatures::RAY_QUERY | 
            gfx::DeviceFeatures::SCALAR_BLOCK_LAYOUT,
        .enable_validation_layer = true,
        //        .enable_gpu_based_validation = true,
    });

    if (result != gfx::Result::SUCCESS) {
        logging::error("raytrace", "Failed to initialize vulkan");
        exit(100);
    }

    gfx::Window window = {};
    result = gfx::CreateWindowWithSwapchain(&window, vk, "raytrace", 1600, 900);
    if (result != gfx::Result::SUCCESS) {
        logging::error("raytrace", "Failed to create vulkan window");
        exit(100);
    }

    VkResult vkr;

    Array<u8> shader_code;
    if (platform::ReadEntireFile("res/raytrace.comp.spirv", &shader_code) != platform::Result::Success) {
        logging::error("raytrace", "Failed to read compute shader");
        exit(100);
    }

    usize total_vertices = 0;
    usize total_indices = 0;
    for(usize i = 0; i < scene.meshes.length; i++) {
        total_vertices += scene.meshes[i].positions.length;
        total_indices += scene.meshes[i].indices.length;
    }
    logging::info("raytrace", "Total indices: %zu | Total vertices: %zu", total_indices, total_vertices);

    gfx::Buffer vertex_buffer = {};
    vkr = gfx::CreateBuffer(&vertex_buffer, vk, total_vertices * sizeof(vec3), {
        .usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        .alloc = gfx::AllocPresets::DeviceMapped,
    });
    if (vkr != VK_SUCCESS) {
        logging::error("raytrace", "Failed to create vertices buffer");
        exit(100);
    }

    gfx::Buffer normals_buffer = {};
    vkr = gfx::CreateBuffer(&normals_buffer, vk, total_vertices * sizeof(vec3), {
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .alloc = gfx::AllocPresets::DeviceMapped,
    });
    if (vkr != VK_SUCCESS) {
        logging::error("raytrace", "Failed to create normals buffer");
        exit(100);
    }

    gfx::Buffer uvs_buffer = {};
    vkr = gfx::CreateBuffer(&uvs_buffer, vk, total_vertices * sizeof(vec2), {
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .alloc = gfx::AllocPresets::DeviceMapped,
    });
    if (vkr != VK_SUCCESS) {
        logging::error("raytrace", "Failed to create uvs buffer");
        exit(100);
    }

    gfx::Buffer index_buffer = {};
    vkr = gfx::CreateBuffer(&index_buffer, vk, total_indices * sizeof(u32), {
        .usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .alloc = gfx::AllocPresets::DeviceMapped,
    });
    if (vkr != VK_SUCCESS) {
        logging::error("raytrace", "Failed to create indices buffer");
        exit(100);
    }

    struct MeshInstance {
        mat4 transform;

        u32 vertex_offset;
        u32 index_offset;
        u32 _padding0;
        u32 _padding1;

        u32 albedo_index;
        u32 normal_index;
        u32 specular_index;
        u32 emissive_index;

        vec4 albedo_value;
        vec4 specular_value;
        vec4 emissive_value;
    };

    gfx::Buffer instances_buffer = {};
    vkr = gfx::CreateBuffer(&instances_buffer, vk, scene.meshes.length * sizeof(MeshInstance), {
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .alloc = gfx::AllocPresets::DeviceMapped,
    });
    if (vkr != VK_SUCCESS) {
        logging::error("raytrace", "Failed to create indices buffer");
        exit(100);
    }

    Array<gfx::AccelerationStructureMeshDesc> meshes(scene.meshes.length);
    usize vertex_offset = 0;
    usize index_offset = 0;

    ArrayView<vec3> vertices = vertex_buffer.map.as_view<vec3>();
    ArrayView<vec3> normals = normals_buffer.map.as_view<vec3>();
    ArrayView<vec2> uvs = uvs_buffer.map.as_view<vec2>();
    ArrayView<u32> indices = index_buffer.map.as_view<u32>();
    ArrayView<MeshInstance> instances = instances_buffer.map.as_view<MeshInstance>();

    VkDeviceAddress vertices_address = gfx::GetBufferAddress(vertex_buffer.buffer, vk.device);
    VkDeviceAddress indices_address = gfx::GetBufferAddress(index_buffer.buffer, vk.device);

    mat4 rotate = glm::rotate(mat4(1.0), 0.75f, vec3(0, 0, 1));

    mat4 to_z_up = mat4(
        vec4(1.f, 0.f, 0.f, 0.f),
        vec4(0.f, 0.f, 1.f, 0.f),
        vec4(0.f, 1.f, 0.f, 0.f),
        vec4(0.f, 0.f, 0.f, 1.f)
    );

    for(usize i = 0; i < scene.meshes.length; i++) {
        usize V = scene.meshes[i].positions.length;
        usize I = scene.meshes[i].indices.length;
        vertices.slice(vertex_offset, V).copy_exact(scene.meshes[i].positions);
        normals.slice(vertex_offset, V).copy_exact(scene.meshes[i].normals);
        uvs.slice(vertex_offset, V).copy_exact(scene.meshes[i].uvs);
        indices.slice(index_offset, I).copy_exact(scene.meshes[i].indices);

        meshes[i] = {
            .vertices_address = vertices_address + vertex_offset * sizeof(vec3),
            .vertices_stride = sizeof(vec3),
            .vertices_count = (u32)V,
            .vertices_format = VK_FORMAT_R32G32B32_SFLOAT,
            .indices_address = indices_address + index_offset * sizeof(u32),
            .indices_type = VK_INDEX_TYPE_UINT32,
            .primitive_count = (u32)(I / 3),
            .transform = rotate * to_z_up * scene.meshes[i].transform,
        };

        Material& mat = scene.meshes[i].material;
        instances[i].vertex_offset = vertex_offset;
        instances[i].index_offset = index_offset;
        instances[i].albedo_index   = mat.base_color.kind == MaterialParameter::Kind::Texture ? mat.base_color.texture   : ~0;
        instances[i].normal_index   = mat.normal.kind     == MaterialParameter::Kind::Texture ? mat.normal.texture       : ~0;
        instances[i].specular_index = mat.specular.kind   == MaterialParameter::Kind::Texture ? mat.specular.texture     : ~0;
        instances[i].emissive_index = mat.emissive.kind   == MaterialParameter::Kind::Texture ? mat.emissive.texture     : ~0;

        instances[i].albedo_value   = mat.base_color.kind != MaterialParameter::Kind::Texture ? mat.base_color.v4   : vec4(0);
        instances[i].specular_value = mat.specular.kind   != MaterialParameter::Kind::Texture ? mat.specular.v4     : vec4(0);
        instances[i].emissive_value = mat.emissive.kind   != MaterialParameter::Kind::Texture ? mat.emissive.v4     : vec4(0);

        vertex_offset += V;
        index_offset += I;
    }


    gfx::AccelerationStructure as = {};
    vkr = gfx::CreateAccelerationStructure(&as, vk, {
        .meshes = Span(meshes),
    });
    if (vkr != VK_SUCCESS) {
        logging::error("raytrace", "Failed to create acceleration structure");
        exit(100);
    }

    Array<gfx::Image> images(scene.images.length);
    for(usize i = 0; i < scene.images.length; i++) {
        VkFormat format = VK_FORMAT_UNDEFINED;
        switch(scene.images[i].format) {
            case Format::RGBA8:      format = VK_FORMAT_R8G8B8A8_UNORM;  break;
            case Format::SRGBA8:     format = VK_FORMAT_R8G8B8A8_SRGB;   break;
            case Format::RGBA8_BC7:  format = VK_FORMAT_BC7_UNORM_BLOCK; break;
            case Format::SRGBA8_BC7: format = VK_FORMAT_BC7_SRGB_BLOCK;  break;
            default: assert(false);
        }

        vkr = gfx::CreateAndUploadImage(&images[i], vk, scene.images[i].data, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, {
            .width = scene.images[i].width,
            .height = scene.images[i].height,
            .format = format,
            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            .alloc = gfx::AllocPresets::Device,
        });
        if (vkr != VK_SUCCESS) {
            logging::error("raytrace", "Failed to create image %zu: %d", i, vkr);
            exit(100);
        }
    }
    gfx::Sampler sampler = {};
    vkr = gfx::CreateSampler(&sampler, vk, {
        .min_filter = VK_FILTER_LINEAR,
        .mag_filter = VK_FILTER_LINEAR,
    });
    if (vkr != VK_SUCCESS) {
        logging::error("raytrace", "Failed to create sampler");
        exit(100);
    }

    gfx::DescriptorSet scene_descriptor_set;
    vkr = gfx::CreateDescriptorSet(&scene_descriptor_set, vk, {
        .entries = {
            {
                .count = 1,
                .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            },
            {
                .count = 1,
                .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            },
            {
                .count = 1,
                .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            },
            {
                .count = 1,
                .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            },
            {
                .count = 1,
                .type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            },
            {
                .count = 1,
                .type = VK_DESCRIPTOR_TYPE_SAMPLER,
            },
            {
                .count = 1,
                .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            },
            {
                .count = (u32)images.length,
                .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            },
        }
    });
    if (vkr != VK_SUCCESS) {
        logging::error("raytrace", "Failed to create scene descriptor set");
        exit(100);
    }

    gfx::WriteBufferDescriptor(scene_descriptor_set.set, vk, {
        .buffer = normals_buffer.buffer,
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .binding = 0,
    });
    gfx::WriteBufferDescriptor(scene_descriptor_set.set, vk, {
        .buffer = uvs_buffer.buffer,
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .binding = 1,
    });
    gfx::WriteBufferDescriptor(scene_descriptor_set.set, vk, {
        .buffer = index_buffer.buffer,
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .binding = 2,
    });
    gfx::WriteBufferDescriptor(scene_descriptor_set.set, vk, {
        .buffer = instances_buffer.buffer,
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .binding = 3,
    });
    gfx::WriteAccelerationStructureDescriptor(scene_descriptor_set.set, vk, {
        .acceleration_structure = as.tlas,
        .binding = 4,
    });
    gfx::WriteSamplerDescriptor(scene_descriptor_set.set, vk, {
        .sampler = sampler.sampler,
        .binding = 5,
    });

    for(usize i = 0; i < images.length; i++) {
        gfx::WriteImageDescriptor(scene_descriptor_set.set, vk, {
            .view = images[i].view,
            .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .binding = 7,
            .element = (u32)i,
        });
    }

    struct Constants {
        u32 width;
        u32 height;
        u32 _padding0;
        u32 _padding1;

        vec3 camera_position;
        u32 _padding2;

        vec3 camera_direction;
        float film_dist;
    };

    Array<gfx::Buffer> constant_buffers(window.frames.length);
    Array<gfx::DescriptorSet> descriptor_sets(window.frames.length);
    for(usize i = 0; i < descriptor_sets.length; i++) {
        vkr = gfx::CreateBuffer(&constant_buffers[i], vk, sizeof(Constants), {
            .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            .alloc = gfx::AllocPresets::DeviceMapped,
        });
        if (vkr != VK_SUCCESS) {
            logging::error("raytrace", "Failed to create uniform buffer");
            exit(100);
        }

        vkr = gfx::CreateDescriptorSet(&descriptor_sets[i], vk, {
            .entries = {
                {
                    .count = 1,
                    .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                },
            }
        });

        if (vkr != VK_SUCCESS) {
            logging::error("raytrace", "Failed to create descriptor set");
            exit(100);
        }

        gfx::WriteBufferDescriptor(descriptor_sets[i].set, vk, {
            .buffer = constant_buffers[0].buffer,
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .binding = 0,
        });
    }


    gfx::Shader shader;
    vkr = gfx::CreateShader(&shader, vk, shader_code);
    if (vkr != VK_SUCCESS) {
        logging::error("raytrace", "Failed to create compute shader");
        exit(100);
    }

    gfx::ComputePipeline compute_pipeline;
    vkr = gfx::CreateComputePipeline(&compute_pipeline, vk, {
        .shader = shader,
        .descriptor_sets = { scene_descriptor_set.layout, descriptor_sets[0].layout },
    });
    if (vkr != VK_SUCCESS) {
        logging::error("raytrace", "Failed to create compute pipeline");
        exit(100);
    }

    struct App {
        // - Window
        bool wait_for_events = false;
        bool closed = false;
        bool first_frame_done = false;

        // - UI
        platform::Timestamp last_frame_timestamp;
        gui::ImGuiImpl gui;

        // - Scene
        vec3 camera_position = vec3(-9.9, -19.044, 4.352);
        vec3 camera_target = vec3(-1., 3., 0.3);
        float film_dist = 0.7;

        // - Rendering
        gfx::AccelerationStructure as;
        Array<gfx::Image> images;
        gfx::Image output_image;
        Array<gfx::Buffer> constant_buffers;
        Array<gfx::DescriptorSet> descriptor_sets;
        gfx::DescriptorSet scene_descriptor_set;
        gfx::ComputePipeline compute_pipeline;
        u32 frame_index = 0; // Rendering frame index, wraps around at the number of frames in flight
    };

    // USER: application
    App app = {};
    app.last_frame_timestamp = platform::GetTimestamp();
    app.compute_pipeline = compute_pipeline;
    app.scene_descriptor_set = scene_descriptor_set;
    app.descriptor_sets = std::move(descriptor_sets);
    app.as = std::move(as);
    app.constant_buffers = std::move(constant_buffers);
    app.images = std::move(images);

    auto MouseMoveEvent = [&app](ivec2 pos) {
    };

    auto MouseButtonEvent = [&app](ivec2 pos, gfx::MouseButton button, gfx::Action action, gfx::Modifiers mods) {
        if (ImGui::GetIO().WantCaptureMouse) return;
    };

    auto MouseScrollEvent = [&app](ivec2 pos, ivec2 scroll) {
        if (ImGui::GetIO().WantCaptureMouse) return;
    };

    auto KeyEvent = [&app, &window](gfx::Key key, gfx::Action action, gfx::Modifiers mods) {
        if (action == gfx::Action::Press || action == gfx::Action::Repeat) {
            if (key == gfx::Key::Escape) {
                gfx::CloseWindow(window);
            }
            if (key == gfx::Key::Comma) {
                app.camera_position.x += 0.05;
                app.camera_position.y -= 0.05;
            }
            if (key == gfx::Key::Period) {
                app.camera_position.x -= 0.05;
                app.camera_position.y += 0.05;
            }
        }
    };

    auto Draw = [&app, &vk, &window]() {
        if (app.closed) return;

        platform::Timestamp timestamp = platform::GetTimestamp();
        float dt = (float)platform::GetElapsed(app.last_frame_timestamp, timestamp);
        app.last_frame_timestamp = timestamp;

        gfx::SwapchainStatus swapchain_status = gfx::UpdateSwapchain(&window, vk);
        if (swapchain_status == gfx::SwapchainStatus::FAILED) {
            logging::error("raytrace/draw", "Swapchain update failed\n");
            exit(101);
        }
        else if (swapchain_status == gfx::SwapchainStatus::MINIMIZED) {
            return;
        }

        if (swapchain_status == gfx::SwapchainStatus::RESIZED || !app.first_frame_done) {
            app.first_frame_done = true;

            gfx::DestroyImage(&app.output_image, vk);
            VkResult vkr = gfx::CreateImage(&app.output_image, vk, {
                .width = window.fb_width,
                .height = window.fb_height,
                .format = VK_FORMAT_R32G32B32A32_SFLOAT,
                .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                .alloc = gfx::AllocPresets::DeviceDedicated,
            });
            if (vkr != VK_SUCCESS) {
                logging::error("raytrace", "Failed to create output image");
                exit(100);
            }
            gfx::WriteImageDescriptor(app.scene_descriptor_set.set, vk, {
                .view = app.output_image.view,
                .layout = VK_IMAGE_LAYOUT_GENERAL,
                .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .binding = 6,
            });

        }

        // Acquire current frame
        gfx::Frame& frame = gfx::WaitForFrame(&window, vk);
        gfx::Result ok = gfx::AcquireImage(&frame, &window, vk);
        if (ok != gfx::Result::SUCCESS) {
            return;
        }


        {
            // USER: pre-gui, but with frame

            {
                gui::BeginFrame();

                // USER: gui
                gui::DrawStats(dt, window.fb_width, window.fb_height);

                if (ImGui::Begin("Editor"))
                {
                    ImGui::DragFloat3("Position", &app.camera_position.x, 0.01);
                    ImGui::DragFloat3("Target", &app.camera_target.x, 0.01);
                }
                ImGui::End();

                // ImGui::ShowDemoWindow();

                gui::EndFrame();
            }


            // Record commands
            gfx::BeginCommands(frame.command_pool, frame.command_buffer, vk);

            gfx::CmdBarriers(frame.command_buffer, {
                .image = {
                    {
                        .src_stage = VK_PIPELINE_STAGE_2_NONE,
                        .src_access = 0,
                        .dst_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        .dst_access = VK_ACCESS_2_SHADER_WRITE_BIT,
                        .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
                        .new_layout = VK_IMAGE_LAYOUT_GENERAL,
                        .image = app.output_image.image,
                    },
                    {
                        .src_stage = VK_PIPELINE_STAGE_2_NONE,
                        .src_access = 0,
                        .dst_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .dst_access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
                        .new_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        .image = frame.current_image,
                    },
                }
            });


            Constants constants;
            constants.width = window.fb_width;
            constants.height = window.fb_height;
            constants.camera_position = app.camera_position;
            constants.camera_direction = glm::normalize(app.camera_target - app.camera_position);
            constants.film_dist = 0.7;

            app.constant_buffers[app.frame_index].map.as_bytes().copy_exact(BytesOf(&constants));

            vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, app.compute_pipeline.pipeline);
            VkDescriptorSet sets[] = {
                app.scene_descriptor_set.set,
                app.descriptor_sets[app.frame_index].set,
            };
            vkCmdBindDescriptorSets(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, app.compute_pipeline.layout, 0, ArrayCount(sets), sets, 0, 0);
            vkCmdDispatch(frame.command_buffer, DivCeil(window.fb_width, 8), DivCeil(window.fb_height, 8), 1);

            gfx::CmdBarriers(frame.command_buffer, {
                .image = {
                    {
                        .src_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        .src_access = VK_ACCESS_2_SHADER_WRITE_BIT,
                        .dst_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .dst_access = VK_ACCESS_2_TRANSFER_READ_BIT,
                        .old_layout = VK_IMAGE_LAYOUT_GENERAL,
                        .new_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        .image = app.output_image.image,
                    },
                }
            });

            VkImageBlit region = {};
            region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.srcSubresource.layerCount = 1;
            region.srcOffsets[1].x = window.fb_width;
            region.srcOffsets[1].y = window.fb_height;
            region.srcOffsets[1].z = 1;
            region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.dstSubresource.layerCount = 1;
            region.dstOffsets[1].x = window.fb_width;
            region.dstOffsets[1].y = window.fb_height;
            region.dstOffsets[1].z = 1;
            vkCmdBlitImage(frame.command_buffer, app.output_image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, frame.current_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_NEAREST);

            gfx::CmdBarriers(frame.command_buffer, {
                .image = {
                    {
                        .src_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .src_access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        .dst_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        .dst_access = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                        .old_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        .image = frame.current_image,
                    },
                }
            });

            gfx::CmdBeginRendering(frame.command_buffer, {
                .color = {
                    {
                        .view = frame.current_image_view,
                        .load_op = VK_ATTACHMENT_LOAD_OP_LOAD,
                        .store_op = VK_ATTACHMENT_STORE_OP_STORE,
                    },
                },
                .width = window.fb_width,
                .height = window.fb_height,
            });

            gui::Render(frame.command_buffer);
            vkCmdEndRenderingKHR(frame.command_buffer);

            gfx::CmdImageBarrier(frame.command_buffer, {
                .src_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .src_access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                .dst_stage = VK_PIPELINE_STAGE_2_NONE,
                .dst_access = 0,
                .old_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .new_layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                .image = frame.current_image,
            });

            gfx::EndCommands(frame.command_buffer);
        }

        VkResult vkr;
        vkr = gfx::Submit(frame, vk, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        assert(vkr == VK_SUCCESS);

        vkr = gfx::PresentFrame(&window, &frame, vk);
        assert(vkr == VK_SUCCESS);

        app.frame_index = (app.frame_index + 1) % (u32)window.frames.length;
    };

    gfx::SetWindowCallbacks(&window, {
        .mouse_move_event = MouseMoveEvent,
        .mouse_button_event = MouseButtonEvent,
        .mouse_scroll_event = MouseScrollEvent,
        .key_event = KeyEvent,
        .draw = Draw,
    });

    gui::CreateImGuiImpl(&app.gui, window, vk, {});

    while (true) {
        gfx::ProcessEvents(app.wait_for_events);

        if (gfx::ShouldClose(window)) {
            logging::info("raytrace", "Window closed");
            app.closed = true;
            break;
        }

        // Draw
        Draw();
    };

    // Wait
    gfx::WaitIdle(vk);

    // USER: cleanup
    gfx::DestroyAccelerationStructure(&app.as, vk);
    gfx::DestroyBuffer(&vertex_buffer, vk);
    gfx::DestroyBuffer(&normals_buffer, vk);
    gfx::DestroyBuffer(&uvs_buffer, vk);
    gfx::DestroyBuffer(&index_buffer, vk);
    gfx::DestroyBuffer(&instances_buffer, vk);
    gfx::DestroySampler(&sampler, vk);
    for(usize i = 0; i < app.images.length; i++) {
        gfx::DestroyImage(&app.images[i], vk);
    }

    gfx::DestroyShader(&shader, vk);
    gfx::DestroyComputePipeline(&app.compute_pipeline, vk);
    gfx::DestroyImage(&app.output_image, vk);
    for(usize i = 0; i < app.descriptor_sets.length; i++) {
        gfx::DestroyBuffer(&app.constant_buffers[i], vk);
        gfx::DestroyDescriptorSet(&app.descriptor_sets[i], vk);
    }
    gfx::DestroyDescriptorSet(&app.scene_descriptor_set, vk);

    // Gui
    gui::DestroyImGuiImpl(&app.gui, vk);

    // Window
    gfx::DestroyWindowWithSwapchain(&window, vk);

    // Context
    gfx::DestroyContext(&vk);
}
