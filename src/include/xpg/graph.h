#pragma once

#include "gfx.h"

namespace framegraph
{
    // Graph
    struct TaskRef {
        static const u32 INVALID_REF = 0xffffffff;
        u32 index = INVALID_REF;

        TaskRef() {}

        explicit TaskRef(usize index) {
            assert(index < TaskRef::INVALID_REF);
            this->index = (u32)index;
        }

        bool valid() {
            return index != TaskRef::INVALID_REF;
        }
    };

    struct ResourceRef {
        TaskRef task;
        u32 index_input_output = 0;

        ResourceRef() {}

        ResourceRef(TaskRef t, usize index, bool is_output, bool is_external) {
            assert(index <= 0x3fffffff);
            task = t;
            index_input_output = (u32)index | ((u32)is_output << 31) | ((u32)is_external << 30);
        }

        bool is_output() {
            return (index_input_output & (1 << 31)) != 0;
        }

        bool is_input() {
            return !is_output();
        }

        bool is_external() {
            return (index_input_output & (1 << 30)) != 0;
        }

        usize index() {
            return index_input_output & 0x3fffffff;
        }

        bool valid() {
            return is_external() || task.valid();
        }
    };

    typedef VkPipelineStageFlags2 ResourceStage;
    typedef VkAccessFlags2 ResourceAccess;
    typedef VkImageLayout ResourceLayout;

    bool has_any_write_access(ResourceAccess flags) {
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

    struct ResourceUsage {
        u32 flags;                  // Either image or buffer usage flags, depending on underlying resource
        ResourceStage first_stage;  // For execution barriers, first stage that r/w to this resource
        ResourceStage last_stage;   // For execution barriers, last stage that r/w to this resource
        ResourceAccess access;      // For memory bariers, all accesses to this resource
        ResourceLayout layout;      // Only used if underlying resource is an image
    };


    namespace RU {
        // TODO: double check that these are fine for Graphics/Compute interop, or for Compute/Compute
        constexpr ResourceUsage Uniform = {
            .flags = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            .first_stage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
            .last_stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .access = VK_ACCESS_2_UNIFORM_READ_BIT,
        };
        constexpr ResourceUsage DrawCommands = {
            .flags = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            .first_stage = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
            .last_stage = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
            .access = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
        };
        constexpr ResourceUsage VertexBuffer = {
            .flags = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            .first_stage = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT,
            .last_stage = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT, // Should this be vertex shader?
            .access = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,
        };

        constexpr ResourceUsage BufferRO{
            .flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .first_stage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
            .last_stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        };
        constexpr ResourceUsage Buffer {
            .flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .first_stage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
            .last_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        };

        constexpr ResourceUsage ImageRO {
            .flags = VK_IMAGE_USAGE_STORAGE_BIT,
            .first_stage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
            .last_stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            .layout = VK_IMAGE_LAYOUT_GENERAL, // This can potentially be READ_ONLY or even SHADER_READ_ONLY?
        };
        constexpr ResourceUsage Image {
            .flags = VK_IMAGE_USAGE_STORAGE_BIT,
            .first_stage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
            .last_stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            .layout = VK_IMAGE_LAYOUT_GENERAL,
        };

        constexpr ResourceUsage RenderTargetWO = {
            .flags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .first_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .last_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };
        constexpr ResourceUsage RenderTarget = {
            .flags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .first_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .last_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .access = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };

        constexpr ResourceUsage DepthStencilTargetRO = {
            .flags = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            .first_stage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
            .last_stage = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            .access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
            .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
        constexpr ResourceUsage DepthStencilTargetWO = {
            .flags = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            .first_stage = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            .last_stage = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            .access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
        constexpr ResourceUsage DepthStencilTarget = {
            .flags = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            .first_stage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
            .last_stage = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            .access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
    }

    enum class ResourceType {
        Buffer,
        Image,
    };

    struct BufferDesc {
        u32 size;
    };

    struct ImageDesc {
        u32 width;
        u32 height;
        VkFormat format;
    };

    // Concrete resource
    struct Resource {
        bool external;
        ResourceType type;
        // Only valid if not external
        union {
            BufferDesc buffer;
            ImageDesc  image;
        };

        // Union of all usages
        u32 usage_flags;

        // State of this resource during resolution
        ResourceStage  current_last_stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        ResourceAccess current_access = VK_ACCESS_2_NONE;
        ResourceLayout current_layout = VK_IMAGE_LAYOUT_UNDEFINED;

        // Last task that accessed this resource. Used to trigger
        // an error if the same task uses the same resource twice.
        TaskRef current_task;
    };

    // Usage of a concrete resource
    struct ResourceHandle {
        static const u32 INVALID_HANDLE = 0xffffffff;
        u32 index = INVALID_HANDLE;

        ResourceHandle() {}

        explicit ResourceHandle(usize index) {
            assert(index < INVALID_HANDLE);
            this->index = (u32)index;
        }

        bool is_valid() {
            return index != INVALID_HANDLE;
        }
    };

    struct ResourceNodeIn {
        TaskRef task;
        ResourceRef edge;

        const char* name; // TODO: update with owned string

        ResourceUsage usage;
        ResourceHandle resource;
    };

    struct ResourceNodeOut {
        TaskRef task;
        Array<ResourceRef> edges;

        const char* name; // TODO: update with owned string

        ResourceHandle resource;
    };

    typedef std::function<void()> Task;

    struct TaskNode {
        Array<ResourceNodeIn> inputs;
        Array<ResourceNodeOut> outputs;

        const char* name; // TODO: update with owned string
        Task task;
    };

    struct Barrier {
        ResourceHandle resource;

        ResourceStage src_stage;
        ResourceAccess src_access;
        ResourceStage dst_stage;
        ResourceAccess dst_access;
        ResourceLayout old_layout;
        ResourceLayout new_layout;
    };

    struct Op {
        // Resources used by this pass
        Array<ResourceHandle> resources;

        // Barriers needed before the task
        Array<Barrier> barriers;

        // Task to be executed
        Task task;
    };

    struct FramePlan {
        // Ordered list of operations to be executed
        Array<Op> ops;

        // List of resources that needs to be created
        Array<Resource> resources;
    };

    // Graph properties:
    // - Two types of nodes:
    //     - tasks: represents some computation (a lambda)
    //     - resources: represents a reference to a resource (+ usage flags / state flags)
    // - Directred edges:
    //     - Always connect a resource to a node or a node to a resource
    //     - Resources only have 1 incoming edge, but many outgoing
    //     - Tasks can have many incoming and outgoing edges

    struct FrameGraph {
        Array<TaskNode> tasks;

        Array<ResourceNodeOut> external;
        Array<ResourceRef> results;

        Array<Resource> resources;

        // Low level graph operations
        TaskRef add_task(const char* name) {
            TaskRef i(tasks.length);
            tasks.add({ .name = name });
            return i;
        }

        ResourceRef add_input(TaskRef t, const char* name) {
            TaskNode& task = get_task(t);
            usize index = task.inputs.length;
            task.inputs.add({
                .task = t,
                .name = name,
            });

            return ResourceRef(t, index, false, false);
        }

        ResourceRef add_output(TaskRef t, const char* name) {
            TaskNode& task = get_task(t);
            usize index = task.outputs.length;
            task.outputs.add({
                .task = t,
                .name = name,
            });

            return ResourceRef(t, index, true, false);
        }

        ResourceRef add_external(const char* name) {
            usize index = external.length;
            ResourceHandle res = add_resource(true);
            external.add({
                .name = name,
                .resource = res,
            });
            return ResourceRef(TaskRef(), index, true, true);
        }

        TaskNode& get_task(TaskRef ref) {
            assert(ref.index != TaskRef::INVALID_REF);
            return tasks[ref.index];
        }

        ResourceNodeIn& get_input_resource(ResourceRef ref) {
            assert(ref.is_input());
            assert(!ref.is_external());
            TaskNode& t = get_task(ref.task);
            return t.inputs[ref.index()];
        }

        ResourceNodeOut& get_output_resource(ResourceRef ref) {
            assert(ref.is_output());
            if (ref.is_external()) {
                return external[ref.index()];
            }
            else {
                TaskNode& t = get_task(ref.task);
                return t.outputs[ref.index()];
            }
        }

        void connect(ResourceRef out, ResourceRef in) {
            ResourceNodeOut& out_node = get_output_resource(out);
            ResourceNodeIn& in_node = get_input_resource(in);

            assert(!in_node.edge.valid());
            out_node.edges.add(in);
            in_node.edge = out;
        }

        // Resources
        ResourceHandle add_resource(bool external) {
            ResourceHandle handle(resources.length);
            resources.add({
                .external = external,
            });
            return handle;
        }

        Resource& get_resource(ResourceHandle handle) {
            assert(handle.is_valid());
            return resources[handle.index];
        }

        // High level graph operations
        ResourceRef read(TaskRef t, ResourceRef r, ResourceUsage usage) {
            ResourceNodeOut& res = get_output_resource(r);
            assert(res.resource.is_valid());

            ResourceRef in = add_input(t, res.name);
            ResourceNodeIn& res_in = get_input_resource(in);
            res_in.usage = usage;
            res_in.resource = res.resource;
            connect(r, in);

            return in;
        }

        ResourceRef write(TaskRef t, ResourceRef r, ResourceUsage usage) {
            read(t, r, usage);

            ResourceNodeOut& res = get_output_resource(r);
            assert(res.resource.is_valid());

            ResourceRef out = add_output(t, res.name);
            ResourceNodeOut& res_out = get_output_resource(out);
            res_out.resource = res.resource;

            return out;
        }

        ResourceRef create_buffer(TaskRef t, const char* name, BufferDesc desc, ResourceUsage usage) {
            ResourceRef ref = add_output(t, name);
            ResourceNodeOut& node = get_output_resource(ref);

            node.resource = add_resource(false);
            Resource& res = get_resource(node.resource);
            res.type = ResourceType::Buffer;
            res.buffer = desc;
            res.usage_flags = usage.flags;
            res.current_access = usage.access;
            res.current_last_stage = usage.last_stage;

            return ref;
        }

        ResourceRef create_image(TaskRef t, const char* name, ImageDesc desc, ResourceUsage usage) {
            ResourceRef ref = add_output(t, name);
            ResourceNodeOut& node = get_output_resource(ref);

            node.resource = add_resource(false);
            Resource& res = get_resource(node.resource);
            res.type = ResourceType::Image;
            res.image = desc;
            res.usage_flags = usage.flags;
            res.current_access = usage.access;
            res.current_last_stage = usage.last_stage;
            res.current_layout = usage.layout;

            return ref;
        }

        void add_result(ResourceRef r) {
            results.add(r);
        }

        // Process
        enum class VisitState {
            Unvisited = 0,
            Visiting = 1,
            Visited = 2,
        };

        void push_ops_task(Array<VisitState>& visit, Array<TaskRef>& ordered_tasks, TaskRef t) {
            if (!t.valid()) return;
            if (visit[t.index] == VisitState::Visited) return;
            if (visit[t.index] == VisitState::Visiting) assert(!"Detected cycle");
            visit[t.index] = VisitState::Visiting;

            TaskNode& task = get_task(t);

            for (usize i = 0; i < task.inputs.length; i++) {
                ResourceNodeIn& in = task.inputs[i];
                ResourceNodeOut& out = get_output_resource(in.edge);
                push_ops_task(visit, ordered_tasks, out.task);
            }

            // printf("%s\n", task.name);
            ordered_tasks.add(t);

            visit[t.index] = VisitState::Visited;
        }

        FramePlan plan() {
            Array<TaskRef> ordered_tasks;
            Array<VisitState> visit(tasks.length);

            // Topological sort and culling starting from desired results
            for (usize i = 0; i < results.length; i++) {
                ResourceNodeOut& ref = get_output_resource(results[i]);
                push_ops_task(visit, ordered_tasks, ref.task);
            }

            // TODO:
            //
            // Instantiate concrete resoruces for the tasks that survived
            // - Task owned resources
            // - External resoruces
            //
            // After this step every task should have a way to
            // reference concrete resources for their inputs.
            //
            // The caller is then responsible for resolving concrete
            // resources during task execution, out of the references
            // stored here.

            Array<Op> ops(ordered_tasks.length);

            for (usize i = 0; i < ordered_tasks.length; i++) {
                TaskRef t = ordered_tasks[i];
                TaskNode& task = get_task(t);


                Array<Barrier> barriers;
                for (usize j = 0; j < task.inputs.length; j++) {
                    ResourceNodeIn& in = task.inputs[j];
                    assert(in.resource.is_valid());

                    // Accumulate resource usages
                    Resource& resource = get_resource(in.resource);
                    resource.usage_flags |= in.usage.flags;

                    // check that the same physical resource is not used twice as a destination for a barrier
                    assert(resource.current_task.index != t.index);

                    // Only need a barrier if the layout is different or if there is a memory barrier needed
                    bool layout_transition = resource.current_layout != in.usage.layout;
                    bool memory_barrier = has_any_write_access(resource.current_access | in.usage.access);
                    if (layout_transition || memory_barrier) {
                        // Add barrier
                        Barrier b = {};
                        b.resource = in.resource;
                        b.old_layout = resource.current_layout;
                        b.new_layout = in.usage.layout;
                        b.src_stage = resource.current_last_stage;
                        b.dst_stage = in.usage.first_stage;
                        b.src_access = resource.current_access;
                        b.dst_access = in.usage.access;
                        barriers.add(b);
                    }

                    // Store transition in concrete resource
                    resource.current_access = in.usage.access;
                    resource.current_last_stage = in.usage.last_stage;
                    resource.current_layout = in.usage.layout;
                    resource.current_task = t;
                }

                ops[i] = {
                    .barriers = move(barriers),
                    .task = move(task.task),
                };
            }

            return {
                .ops = move(ops),
            };
        }

        // Debug

        void dump_dot(const char* path) {
            FILE* f = fopen(path, "wb");
            fprintf(f, "digraph RenderGraph {\n");
            fprintf(f, "graph [ ranksep = 2; rankdir = LR; ]\n");

            // Edges
            for (usize i = 0; i < tasks.length; i++) {
                TaskNode& t = tasks[i];
                for (usize j = 0; j < t.outputs.length; j++) {
                    ResourceNodeOut& out = t.outputs[j];
                    for (usize k = 0; k < out.edges.length; k++) {
                        ResourceRef edge = out.edges[k];
                        ResourceNodeIn& in = get_input_resource(edge);
                        TaskNode& other = get_task(in.task);
                        fprintf(f, "%s:%s_out:e -> %s:%s_in:w;\n", t.name, out.name, other.name, in.name);
                    }
                }
            }

            // Graph parameter edges
            for (usize i = 0; i < external.length; i++) {
                ResourceNodeOut& out = external[i];
                for (usize k = 0; k < out.edges.length; k++) {
                    ResourceRef edge = out.edges[k];
                    ResourceNodeIn& in = get_input_resource(edge);
                    TaskNode& other = get_task(in.task);
                    fprintf(f, "%s_param:e -> %s:%s_in:w;\n", out.name, other.name, in.name);
                }
            }

            // Nodes
            fprintf(f, "subgraph nodes {\n");
            for (usize i = 0; i < tasks.length; i++) {
                TaskNode& t = tasks[i];

                fprintf(f, "%s [ shape = record, label= \"", t.name);

                fprintf(f, "{{");
                for (usize j = 0; j < t.inputs.length; j++) {
                    ResourceNodeIn& r = t.inputs[j];
                    fprintf(f, "<%s_in> %s", r.name, r.name);
                    if (j != t.inputs.length - 1) {
                        fprintf(f, "|");
                    }
                }
                fprintf(f, "}");
                fprintf(f, "|%s|", t.name);
                fprintf(f, "{");
                for (usize j = 0; j < t.outputs.length; j++) {
                    ResourceNodeOut& r = t.outputs[j];
                    fprintf(f, "{<%s_out> %s}", r.name, r.name);
                    if (j != t.outputs.length - 1) {
                        fprintf(f, "|");
                    }
                }
                fprintf(f, "}}\" ];\n");
            }

            // Graph parameter nodes
            for (usize i = 0; i < external.length; i++) {
                ResourceNodeOut& out = external[i];
                fprintf(f, "%s_param [ shape = record, label=\"%s\" ];", out.name, out.name);
            }

            fprintf(f, "}\n");
            fprintf(f, "}\n");
            fclose(f);
        }
    };

#if 0
    void test() {
        // TODO:
        //
        // | On graph only (FrameGraph API) |
        //
        // Define graph: (also from python)
        // - create / read / write resources
        // - import / export resources
        // - define accesses of each resources
        // - define lambdas
        //
        // Process graph into plan:
        // - validation
        // - culling
        // - linearization -> barriers + tasks
        //
        //
        // | Requires vulkan context (Passed in through lambdas) |
        //
        // Setup:
        // - pass in external resources
        // - transient resource creation
        //
        // Execute:
        // - lookup resources   -> returns either transient or external physical resource previously created
        // - put barriers       -> iterate over list of barrier, converting resource handle to physical resource. Use all this information to push commands
        // - write descriptors  -> ???
        // - dispatch / draw    -> execute commands
        //
        //
        // | Notes: |
        //
        // Ideally the rendering would be completely decoupled from the graph creation.
        // The only real link is the FramePlan (that contains the ordering of tasks and barriers)
        // and the ResourceRef that are used both at graph creation time and then again
        // at graph execution time to resolve physical resources.
        //
        // In theory internal / external resource creation can be deferred to any later stage
        // and the resource lookup logic can be user defined, the framegraph only works with
        // handles.
        //
        // Currently we have 2 notions of handles in the graph, ResourceRef and ResourceHandle,
        // ResourceRef is a reference to a node in the graph, multiple nodes can have the same
        // ResourceHandle. ResourceHandles point to concrete resources, they can be external
        // or owned by the task.
        // We can got from ResourceRef to physical resource with a double deref, its
        // not clear if we could potentially do this in a single lookup.
        //
        // In the current system we create resource handles at external resource creation time
        // and at owned resource creation time, and then copy those along while creating the graph.
        // This means if the graph is culled during planning, some of the resource handles are not
        // interesting anymore and potentially we could throw them away.
        // We could fix this by either replacing the handles at resolution time or just live
        // with the fact that there can be holes in the resource array.
        //
        //
        // Resource resolution:
        //
        // After planning we need to create concrete resources for each resource handle.
        // For owned resources we have creation information in the Resource class.
        // For external resources we expect creation to already have happened.
        //
        // During rendering the task callback is called with ResourceRef, we want to probably
        // allow the user to specify a custom resolution mechanism for external resources,
        // this could be as simple as a function that uses an hash table to map from
        // ResourceHandle to a ConcreteResource type.
        //
        // A ConcreteResource needs to contain all the data required by a task to do its work,
        // this should include the following:
        // - VkImage or VkBuffer
        // - Descriptors
        //
        // A task also needs pipelines and potentially other global information. Some of this
        // could even just be captured by the lambda if defined inline with the application.
        //
        // If tasks are supposed to be completely self contained then they probably also need
        // a common way to define pipelines and parameters. And retrieve this data during
        // execution.
        //
        // TODO:
        // [ ] Sketch out how some sample applications with a render graph could look like:
        //     e.g.:
        //      - simple shadow map + opaque draw
        //      - bindless rendering example
        //      - streaming data from disk (e.g. sequence / bigimage)
        //      - library of multiple reusable passes (e.g. gbufferf, ddgi, postprocess, shadows, SSAO, SSR)
        // [ ] Figure out what can be a convenient way to use this from python.
        //     - graph creation based on python + shader reflection
        //     - graph execution in python callbacks with helpers in C++ that can call callable objects in python
        // [ ] See if we can somehow leverage shader reflection to easen creation of render graphs
        //     technically the shader knows quite a bit about which resources are used how.
        //     (Still generally does not have complete information, e.g. render target / texture formats)
        // [ ] See what is a conventient bindind model, if should have the application handle this completely
        //     and just hand out helpers, or if the framegraph can also help with this.
        //
        // [ ] Use all this information to improve the design of the framegraph


        struct External {
            ResourceRef positions;
            ResourceRef constants;
            ResourceRef commands;
            ResourceRef backbuffer;

            External(FrameGraph& g) {
                this->positions = g.add_external("positions");
                this->constants = g.add_external("constants");
                this->commands = g.add_external("commands");
                this->backbuffer = g.add_external("backbuffer");
            }
        };

        struct DrawOpaque {
            // Inputs
            ResourceRef commands;
            ResourceRef constants;
            ResourceRef positions;

            // Outputs
            ResourceRef feedback;
            ResourceRef color;
            ResourceRef normal_material;
            ResourceRef depth;
        };

        FrameGraph g;

        DrawOpaque draw_opaque = {};
        draw_opaque.commands = g.read(t, commands, RU::DrawCommands);
        draw_opaque.constants = g.read(t, constants, RU::Uniform);
        draw_opaque.positions = g.read(t, positions, RU::VertexBuffer);
        draw_opaque.feedback = g.create_buffer(t, "feedback", { .size = 1024 }, RU::Buffer);
        draw_opaque.color = g.create_image(t, "color", { .width = 1920, .height = 1080, .format = VK_FORMAT_R16G16B16A16_SFLOAT }, RU::RenderTarget);
        draw_opaque.normal_material = g.create_image(t, "normal_material", { .width = 1920, .height = 1080, .format = VK_FORMAT_R16G16B16A16_SFLOAT }, RU::RenderTarget);
        draw_opaque.depth = g.create_image(t, "depth", { .width = 1920, .height = 1080, .format = VK_FORMAT_R16G16B16A16_SFLOAT }, RU::DepthStencilTarget);

        g.addTask("DrawOpaque",
            [&](ResourcePool& resources) {
                gfx:Buffer img = resources.getBuffer(draw_opaque.commands);
            },
        });

        // DrawOpaque& draw_opaque = g.addTask<DrawOpaque>("DrawOpaque",
        //     [&](FrameGraph& g, TaskRef t, DrawOpaque& data) {
        //         // Inputs
        //         data.commands = g.read(t, commands, RU::DrawCommands);
        //         data.constants = g.read(t, constants, RU::Uniform);
        //         data.positions = g.read(t, positions, RU::VertexBuffer);

        //         // Outputs
        //         data.feedback = g.create_buffer(t, "feedback", { .size = 1024 }, RU::Buffer);
        //         data.color = g.create_image(t, "color", { .width = 1920, .height = 1080, .format = VK_FORMAT_R16G16B16A16_SFLOAT }, RU::RenderTarget);
        //         data.normal_material = g.create_image(t, "normal_material", { .width = 1920, .height = 1080, .format = VK_FORMAT_R16G16B16A16_SFLOAT }, RU::RenderTarget);
        //         data.depth = g.create_image(t, "depth", { .width = 1920, .height = 1080, .format = VK_FORMAT_R16G16B16A16_SFLOAT }, RU::DepthStencilTarget);
        //     },
        //     [](gfx::Context& vk, DrawOpaque& data, ResourcePool& resources) {
        //         gfx:Buffer img = resources.getBuffer(data.commands);
        //     },
        // });

        struct GBuffer {
            ResourceRef color;

            GBuffer(FrameGraph& g,
                    ResourceRef constants,
                    ResourceRef color,
                    ResourceRef normal_material,
                    ResourceRef depth)
            {
                TaskRef t = g.add_task("GBuffer");

                g.read(t, constants, RU::Uniform);
                g.read(t, color, RU::ImageRO);
                g.read(t, normal_material, RU::ImageRO);
                g.read(t, depth, RU::ImageRO);

                this->color = g.create_image(t, "color", { .width = 1920, .height = 1080, .format = VK_FORMAT_R16G16B16A16_SFLOAT }, RU::Image);
            }
        };

        struct Tonemap {
            ResourceRef out;
            Tonemap(FrameGraph& g, ResourceRef in, ResourceRef out) {
                TaskRef t = g.add_task("Tonemap");

                g.read(t, in, RU::ImageRO);
                this->out = g.write(t, out, RU::Image);

            }
        };

        // Definition
        External external(g);
        DrawOpaque draw_opaque(g, external.commands, external.constants, external.positions);
        GBuffer gbuffer(g, external.constants, draw_opaque.color, draw_opaque.normal_material, draw_opaque.depth);
        Tonemap tonemap(g, gbuffer.color, external.backbuffer);

        g.add_result(tonemap.out);

        g.dump_dot("test.dot");
        FramePlan plan = g.plan();
#if 0
        // Process
        FramePlan plan = g.plan();

        // Setup
        gfx::Context vk = {};
        plan.create_transient_resources(vk);

        while (true) {
            // TODO: swapchain acquire

            // TODO: handle resize

            // TODO: update external resources

            // Execute (can be extracted to plan.render_frame()?)
            for (usize i = 0; i < plan.ops.length; i++) {
                Op& op = plan.ops[i];

                for (usize j = 0; j < op.barriers.length; j++) {
                    Barrier& b = op.barriers[j];

                    // TODO: Execute barrier
                }

                // Execute task
                op.task(vk)
            }

            // TODO: present to swapchain
        }
#endif

#if 0
        TaskRef ta = g.add_task("Animate");
        g.add_input(ta, "Vr");
        ResourceRef v_out = g.add_output(ta, "V");

        TaskRef t0 = g.add_task("Shadows");
        ResourceRef shadow_v_in = g.add_input(t0, "V");
        g.add_input(t0, "I");
        ResourceRef shadow_map_out = g.add_output(t0, "ShadowMap");

        TaskRef t1 = g.add_task("Opaque");
        ResourceRef opaque_v_in = g.add_input(t1, "V");
        g.add_input(t1, "I");
        g.add_input(t1, "RenderTarget");
        ResourceRef shadow_map_in = g.add_input(t1, "ShadowMap");
        ResourceRef opaque_rt_out = g.add_output(t1, "RenderTarget");

        TaskRef t2 = g.add_task("PostProcess");
        ResourceRef post_rt_in = g.add_input(t2, "RenderTarget");
        g.add_output(t2, "RenderTarget");

        g.connect(v_out, shadow_v_in);
        g.connect(v_out, opaque_v_in);

        g.connect(shadow_map_out, shadow_map_in);
        g.connect(opaque_rt_out, post_rt_in);

        g.dump_dot("test.dot");

        FramePlan plan = g.plan();
        for (usize i = 0; i < plan.ops.length; i++) {
            Op& op = plan.ops[i];

            for (usize j = 0; j < op.barriers.length; j++) {
                Barrier& b = op.barriers[j];

                // Execute barrier
            }

            // Execute task
        }
#endif
    }
}
#endif