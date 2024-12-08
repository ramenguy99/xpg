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

        bool is_valid() {
            return index != TaskRef::INVALID_REF;
        }
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

    typedef VkPipelineStageFlags2 ResourceStage;
    typedef VkAccessFlags2 ResourceAccess;
    typedef VkImageLayout ResourceLayout;

    struct ResourceState {
        ResourceLayout layout;
        ResourceStage stage;
        ResourceAccess access;
    };

    struct ResourceDesc {
        u32 width;
        u32 height;
        VkFormat format;
    };

    struct Resource {
        const char* name;
        ResourceDesc desc;
    };

    // Concrete resource
    struct ResourceUsage {
        // Task that uses this resource (target of the edge)
        TaskRef task;

        // State required for the usage of the resource
        ResourceState state;
    };

    struct ResourceNode {
        TaskRef parent;
        Array<TaskRef> users;
        
        // Handle to a real resource that is populated after resolving
        ResourceHandle resource;
    };

    typedef std::function<void()> Task;
    struct TaskNode {
        const char* name;
        Array<ResourceUsage> inputs;
        Array<ResourceRef> outputs;
        Task task;
    };

    struct Barrier {
        ResourceHandle resource;
        ResourceState src;
        ResourceState dst;
    };

    struct Op {
        // Barriers needed before the task
        Array<Barrier> barriers;

        // Task to be executed
        TaskRef task;
    };

    struct FramePlan {
        // Ordered list of operations to be executed
        Array<Op> ops;
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
        Array<TaskNode> task_nodes;
        Array<ResourceNode> resource_nodes;

        Array<ResourceNode> external;

        Array<Resource> concrete_resources;

        // Low level graph operations
        TaskRef add_task(const char* name) {
            TaskRef i(tasks.length);
            tasks.add({ .name = name });
            return i;
        }

        ResourceN add_resource(const char* name) {
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
        ResourceRef read(TaskRef t, ResourceRef r, ResourceState state) {
            ResourceNodeOut& res = get_output_resource(r);
            assert(res.resource.is_valid());

            ResourceRef in = add_input(t, res.name);
            ResourceNodeIn& res_in = get_input_resource(in);
            res_in.state = state;
            res_in.resource = res.resource;
            connect(r, in);

            return in;
        }

        ResourceRef write(TaskRef t, ResourceRef r, ResourceState state) {
            read(t, r, state);

            ResourceNodeOut& res = get_output_resource(r);
            assert(res.resource.is_valid());
            get_resource(res.resource).dirty = true;

            ResourceRef out = add_output(t, res.name);
            get_output_resource(out).resource = res.resource;

            return out;
        }

        ResourceRef create(TaskRef t, const char* name, ResourceDesc res) {
            ResourceRef ref = add_output(t, name);
            ResourceNodeOut& node = get_output_resource(ref);
            node.resource = add_resource(false);
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

        void push_ops_task(Array<VisitState>& visit, Array<Op>& ops, TaskRef t) {
            if (!t.valid()) return;
            if (visit[t.index] == VisitState::Visited) return;
            if (visit[t.index] == VisitState::Visiting) assert(!"Detected cycle");
            visit[t.index] = VisitState::Visiting;

            TaskNode& task = get_task(t);

            for (usize i = 0; i < task.inputs.length; i++) {
                ResourceNodeIn& in = task.inputs[i];
                ResourceNodeOut& out = get_output_resource(in.edge);
                push_ops_task(visit, ops, out.task);
            }

            // printf("%s\n", task.name);
            ops.add({
                .task = task.task,
            });

            visit[t.index] = VisitState::Visited;
        }

        FramePlan plan() {
            Array<Op> ops;
            Array<VisitState> visit(tasks.length);

            // Topological sort and culling starting from desired results
            for (usize i = 0; i < results.length; i++) {
                ResourceNodeOut& ref = get_output_resource(results[i]);
                push_ops_task(visit, ops, ref.task);
            }

            // TODO: Add barriers looking at before / after state

            // TODO: check that the same resource is not used twice as a destination for a barrier (somehow mark then unmark just for this set of barriers?)

            return {
                .ops = std::move(ops),
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
        // - lookup resources
        // - put barriers
        // - write descriptors
        // - dispatch / draw

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
            ResourceRef color;
            ResourceRef normal_material;
            ResourceRef depth;

            DrawOpaque(FrameGraph& g, 
                       ResourceRef commands,
                       ResourceRef constants,
                       ResourceRef positions
            ) {
                TaskRef t = g.add_task("DrawOpaque");
                g.read(t, commands, {});
                g.read(t, constants, {});
                g.read(t, positions, {});

                ResourceDesc desc = {};
                this->color = g.create(t, "color", desc);
                this->normal_material = g.create(t, "normal_material", desc);
                this->depth = g.create(t, "depth", desc);
            }
        };

        struct GBuffer {
            ResourceRef color;

            GBuffer(FrameGraph& g, 
                    ResourceRef constants,
                    ResourceRef color,
                    ResourceRef normal_material,
                    ResourceRef depth)
            {
                TaskRef t = g.add_task("GBuffer");

                g.read(t, constants, {});
                g.read(t, color, {});
                g.read(t, normal_material, {});
                g.read(t, depth, {});

                ResourceDesc desc = {};
                this->color = g.create(t, "color", desc);
            }
        };

        struct Tonemap {
            ResourceRef out;
            Tonemap(FrameGraph& g, ResourceRef in, ResourceRef out) {
                TaskRef t = g.add_task("Tonemap");

                g.read(t, in, {});
                this->out = g.write(t, out, {});
            }
        };

        // Definition
        FrameGraph g;
        External external(g);
        DrawOpaque draw_opaque(g, external.commands, external.constants, external.positions);
        GBuffer gbuffer(g, external.constants, draw_opaque.color, draw_opaque.normal_material, draw_opaque.depth);
        Tonemap tonemap(g, gbuffer.color, external.backbuffer);

        g.add_result(tonemap.out);

        g.dump_dot("test.dot");
        g.plan();
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
