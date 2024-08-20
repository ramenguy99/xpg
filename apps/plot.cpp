#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utility> // std::move
#include <functional> // std::function
#include <mutex>
#include <unordered_map>

#ifdef _WIN32
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <semaphore.h>
#include <pthread.h>
#endif

#define VOLK_IMPLEMENTATION
#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>

#define VMA_STATIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#define _GLFW_VULKAN_STATIC
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#endif
#include <GLFW/glfw3.h>


#undef APIENTRY
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui.h>
#include <imgui.cpp>
#include <imgui_demo.cpp>
#include <imgui_draw.cpp>
#include <imgui_tables.cpp>
#include <imgui_widgets.cpp>

#include <implot.h>
#include <implot_internal.h>
#include <implot.cpp>
#include <implot_items.cpp>
#include <implot_demo.cpp>

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_glfw.cpp>

#undef VK_NO_PROTOTYPES
#include <backends/imgui_impl_vulkan.h>
#include <backends/imgui_impl_vulkan.cpp>

#include <atomic_queue/atomic_queue.h>

#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <CLI11.hpp>

#define XPG_VERSION 0

#include "defines.h"
#include "array.h"
#include "platform.h"
#include "threading.h"
#include "gfx.h"
#include "buffered_stream.h"

#define SPECTRUM_USE_DARK_THEME
#include "imgui_spectrum.h"
#include "roboto-medium.h"

using glm::vec3;
using glm::mat4;
#include "types.h"

enum class BinaryType {
    None,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
    f32,
    f64,
    Void, // Empty type not considered for plotting, useful to deal with padding / undesired fields.
};

struct BinaryFormat {
    u64 offset;
    u64 size;
    BinaryType type;
};

struct Range {
    s32 begin;
    s32 end;
};

struct Color {
    ImU32 color;
};

struct Lines {
    s32 x;
    std::vector<Range> ys;
};

struct AxisLimits {
    bool set = false;
    double min;
    double max;
};

enum class PlotKind {
    Plot,
    Histogram,
};

struct Plot {
    PlotKind kind = PlotKind::Plot;

    std::vector<Lines> lines;

    std::vector<ImPlotMarker_> markers;
    std::vector<Color> colors;
    std::vector<std::string> names;
    double size = -1.0;
    double width = -1.0;
    bool grid = true;
    std::string title;
    std::string xlabel;
    std::string ylabel;
    ImPlotScale_ xscale = ImPlotScale_Linear;
    ImPlotScale_ yscale = ImPlotScale_Linear;
    AxisLimits xlimits;
    AxisLimits ylimits;

    // Histogram specific
    int histogram_bins = ImPlotBin_Sturges;
    ImPlotHistogramFlags histogram_flags;
    AxisLimits histogram_range;
};

struct Data {
    Array<double> data;
    s64 columns = 0;
};

struct App {
    VulkanContext* vk;
    VulkanWindow* window;

    VkQueue queue;

    // Swapchain frames, index wraps around at the number of frames in flight.
    u32 frame_index;
    Array<VulkanFrame> frames;
    Array<VkFramebuffer> framebuffers; // Currently outside VulkanFrame because it's not needed when using Dyanmic rendering.

    // Total frame index.
    u64 current_frame;

    bool force_swapchain_update;
    bool wait_for_events;
    bool closed;

    //- Application data.
    u64 last_frame_timestamp;

    std::mutex mutex;
    Data data;
    std::vector<Plot> plots;
    VkRenderPass imgui_render_pass;
};

std::string StringPrintf(const char* format, ...) {
    // Compute size
    va_list args;
    va_start(args, format);
    size_t size = vsnprintf(nullptr, 0, format, args) + 1;
    va_end(args);

    // Alloc string and print into it
    std::string str;
    str.resize(size);
    va_start(args, format);
    vsnprintf(&str[0], size, format, args);
    va_end(args);

    // remove trailing NUL
    str.pop_back();

    return str;
}
void Draw(App* app) {
    if (app->closed) return;

    auto& vk = *app->vk;
    auto& window = *app->window;

    u64 timestamp = glfwGetTimerValue();

    float dt = (float)((double)(timestamp - app->last_frame_timestamp) / (double)glfwGetTimerFrequency());

    SwapchainStatus swapchain_status = UpdateSwapchain(&window, vk, app->force_swapchain_update);
    if (swapchain_status == SwapchainStatus::FAILED) {
        printf("Swapchain update failed\n");
        exit(1);
    }
    app->force_swapchain_update = false;

    if (swapchain_status == SwapchainStatus::MINIMIZED) {
        // app->wait_for_events = true;
        return;
    }
    else if(swapchain_status == SwapchainStatus::RESIZED) {
        // Resize framebuffer sized elements.
        for(usize i = 0; i < app->framebuffers.length; i++) {
            vkDestroyFramebuffer(app->vk->device, app->framebuffers[i], 0);
            app->framebuffers[i] = VK_NULL_HANDLE;
        }
    }

    for(usize i = 0; i < app->framebuffers.length; i++) {
        if(app->framebuffers[i] != VK_NULL_HANDLE) continue;

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = app->imgui_render_pass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = &window.image_views[i];
        framebufferInfo.width = window.fb_width;
        framebufferInfo.height = window.fb_height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(app->vk->device, &framebufferInfo, nullptr, &app->framebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
    // app->wait_for_events = false;

    // Acquire current frame
    VulkanFrame& frame = app->frames[app->frame_index];

    vkWaitForFences(vk.device, 1, &frame.fence, VK_TRUE, ~0);

    u32 index;
    VkResult vkr = vkAcquireNextImageKHR(vk.device, window.swapchain, ~0ull, frame.acquire_semaphore, 0, &index);
    if(vkr == VK_ERROR_OUT_OF_DATE_KHR) {
        app->force_swapchain_update = true;
        return;
    }

    // ImGui
    ImGui_ImplGlfw_NewFrame();
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();

    // ImGuiID dockspace = ImGui::DockSpaceOverViewport(NULL, ImGuiDockNodeFlags_PassthruCentralNode);
    // static bool first_frame = true;

    // ImGui::ShowDemoWindow();
    ImPlot::ShowDemoWindow();
    {
        std::lock_guard<std::mutex> lock(app->mutex);

        usize count = app->data.columns > 0 ? app->data.data.length / app->data.columns : 0;
        usize stride = app->data.columns * sizeof(double);

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        if(ImGui::Begin("Window###window", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove)) {
            int plots_per_axis = (int)sqrt(app->plots.size());
            int plots_x = ((int)app->plots.size() + plots_per_axis - 1) / plots_per_axis;
            int plots_y = ((int)app->plots.size() + plots_per_axis - 1) / plots_x;
            if(ImPlot::BeginSubplots("###subplots", plots_y, plots_x, ImVec2(-1, -1))) {
                for(usize plot_idx = 0; plot_idx < app->plots.size(); plot_idx++) {
                    auto& plot = app->plots[plot_idx];
                    std::string title;
                    if(plot.title.empty()) {
                        title = StringPrintf("Plot %d###plot%d", (int)plot_idx, (int)plot_idx);
                    } else {
                        title = StringPrintf("%s###plot%d", plot.title.c_str(), (int)plot_idx);
                    }
                    if(ImPlot::BeginPlot(title.c_str(), ImVec2(-1, -1))) {

                        ImPlotAxisFlags xaxis_flags = 0;
                        ImPlotAxisFlags yaxis_flags = 0;
                        if(!plot.grid) {
                            xaxis_flags |= ImPlotAxisFlags_NoGridLines;
                            yaxis_flags |= ImPlotAxisFlags_NoGridLines;
                        }

                        ImPlot::SetupAxis(ImAxis_X1, plot.xlabel.c_str(), xaxis_flags);
                        ImPlot::SetupAxis(ImAxis_Y1, plot.ylabel.c_str(), yaxis_flags);

                        if(plot.xscale != ImPlotScale_Linear) { ImPlot::SetupAxisScale(ImAxis_X1, plot.xscale); }
                        if(plot.yscale != ImPlotScale_Linear) { ImPlot::SetupAxisScale(ImAxis_Y1, plot.yscale); }

                        if(plot.xlimits.set) { ImPlot::SetupAxisLimits(ImAxis_X1, plot.xlimits.min, plot.xlimits.max); }
                        if(plot.ylimits.set) { ImPlot::SetupAxisLimits(ImAxis_Y1, plot.ylimits.min, plot.ylimits.max); }

                        u64 per_plot_style_vars = 0;
                        if(plot.size >= 0.0) {
                            ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, (float)plot.size);
                            per_plot_style_vars += 1;
                        }
                        if(plot.width >= 0.0) {
                            ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, (float)plot.width);
                            per_plot_style_vars += 1;
                        }

                        u64 y_idx = 0;
                        for(auto& line: plot.lines) {
                            int x = line.x;
                            for(auto& range: line.ys) {
                                for(s32 y = range.begin; y <= std::min(range.end, (s32)app->data.columns - 1); y++) {
                                    u64 per_line_style_vars = 0;
                                    if(!plot.markers.empty()) {
                                        ImPlot::PushStyleVar(ImPlotStyleVar_Marker, plot.markers[y_idx % plot.markers.size()]);
                                        per_line_style_vars += 1;
                                    }

                                    u64 per_line_colors = 0;
                                    if(!plot.colors.empty()) {
                                        ImPlot::PushStyleColor(ImPlotCol_Line, plot.colors[y_idx % plot.colors.size()].color);
                                        ImPlot::PushStyleColor(ImPlotCol_Fill, plot.colors[y_idx % plot.colors.size()].color);
                                        per_line_colors += 2;
                                    }
                                    ImPlotFlags flags = 0;
                                    std::string name;
                                    if(plot.names.empty()) {
                                        name = StringPrintf("%d###line%d", (int)y_idx, (int)y_idx);
                                    } else {
                                        name = StringPrintf("%d - %s###line%d", (int)y_idx, plot.names[y_idx % plot.names.size()].c_str(), (int)y_idx);
                                    }

                                    if(plot.kind == PlotKind::Plot) {
                                        if(x != -1) {
                                            ImPlot::GetterXY<ImPlot::IndexerIdx<double>, ImPlot::IndexerIdx<double>> getter(
                                                ImPlot::IndexerIdx<double>(app->data.data.data + x, (int)count, 0, (int)stride),
                                                ImPlot::IndexerIdx<double>(app->data.data.data + y, (int)count, 0, (int)stride),
                                                (int)count
                                            );
                                            ImPlot::PlotLineEx(name.c_str(), getter, flags);
                                        } else {
                                            ImPlot::GetterXY<ImPlot::IndexerLin, ImPlot::IndexerIdx<double>> getter(
                                                ImPlot::IndexerLin(1, 0),
                                                ImPlot::IndexerIdx<double>(app->data.data.data + y, (int)count, 0, (int)stride),
                                                (int)count
                                            );
                                            ImPlot::PlotLineEx(name.c_str(), getter, flags);
                                        }
                                    } else {
                                        ImPlotRange range = ImPlotRange();
                                        if(plot.histogram_range.set) {
                                            range = ImPlotRange(plot.histogram_range.min, plot.histogram_range.max);
                                        }
                                        Array<double> values(count);
                                        for(usize i = 0; i < count; i++) {
                                            values[i] = app->data.data.data[i * app->data.columns + y];
                                        }
                                        ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL,0.5f);
                                        ImPlot::PlotHistogram(name.c_str(), values.data, (int)count, plot.histogram_bins, 1.0, range, plot.histogram_flags);
                                    }

                                    y_idx += 1;
                                    for(u64 i = 0; i < per_line_style_vars; i++) {
                                        ImPlot::PopStyleVar();
                                    }
                                }
                            }
                        }

                        for(u64 i = 0; i < per_plot_style_vars; i++) {
                            ImPlot::PopStyleVar();
                        }
                        ImPlot::EndPlot();
                    }
                }
                ImPlot::EndSubplots();
            }
        }
        ImGui::End();
    }

    // Render imgui.
    ImGui::Render();

    // Reset command pool
    vkr = vkResetCommandPool(vk.device, frame.command_pool, 0);
    assert(vkr == VK_SUCCESS);

    // Record commands
    VkCommandBufferBeginInfo begin_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkr = vkBeginCommandBuffer(frame.command_buffer, &begin_info);
    assert(vkr == VK_SUCCESS);

    vkResetFences(vk.device, 1, &frame.fence);

    VkClearColorValue color = { 0.1f, 0.2f, 0.4f, 1.0f };

    VkClearValue clear_values[1];
    clear_values[0].color = color;

    VkRenderPassBeginInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    info.renderPass = app->imgui_render_pass;
    info.framebuffer = app->framebuffers[index];
    info.renderArea.extent.width = window.fb_width;
    info.renderArea.extent.height = window.fb_height;
    info.clearValueCount = ArrayCount(clear_values);
    info.pClearValues = clear_values;
    vkCmdBeginRenderPass(frame.command_buffer, &info, VK_SUBPASS_CONTENTS_INLINE);

    ImDrawData* draw_data = ImGui::GetDrawData();
    ImGui_ImplVulkan_RenderDrawData(draw_data, frame.command_buffer);

    vkCmdEndRenderPass(frame.command_buffer);

    vkr = vkEndCommandBuffer(frame.command_buffer);
    assert(vkr == VK_SUCCESS);

    // Submit commands
    VkPipelineStageFlags submit_stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSubmitInfo submit_info = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &frame.command_buffer;
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = &frame.acquire_semaphore;
    submit_info.pWaitDstStageMask = &submit_stage_mask;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &frame.release_semaphore;


    vkr = vkQueueSubmit(app->queue, 1, &submit_info, frame.fence);
    assert(vkr == VK_SUCCESS);

    // Present
    VkPresentInfoKHR present_info = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &window.swapchain;
    present_info.pImageIndices = &index;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = &frame.release_semaphore;
    vkr = vkQueuePresentKHR(app->queue, &present_info);

    if (vkr == VK_ERROR_OUT_OF_DATE_KHR || vkr == VK_SUBOPTIMAL_KHR) {
        app->force_swapchain_update = true;
    } else if (vkr != VK_SUCCESS) {
        printf("Failed to submit\n");
        exit(1);
    }

    app->frame_index = (app->frame_index + 1) % window.images.length;
    app->current_frame += 1;
}

internal void
Callback_Key(GLFWwindow* window, int key, int scancode, int action, int mods) {
    App* app = (App*)glfwGetWindowUserPointer(window);

    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        if (key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(app->window->window, true);
        }
    }
}

internal void
Callback_WindowResize(GLFWwindow* window, int width, int height) {
    App* app = (App*)glfwGetWindowUserPointer(window);
}

internal void
Callback_WindowRefresh(GLFWwindow* window) {
    App* app = (App*)glfwGetWindowUserPointer(window);
    if (app) {
        Draw(app);
    }
}

#ifdef _WIN32
DWORD WINAPI thread_proc(void* param) {
    HWND window = (HWND)param;
    while (true) {
        SendMessage(window, WM_PAINT, 0, 0);
    }
    return 0;
}
#endif

// TODO:
// - Input types:
//   [x] File
//   [x] Stdin
// - Plot types:
//   [x] Normal 2D plot (y only, multiple ys, one x and one or more ys, x, y pairs)
//   [x] Histogram (configurable bin size) and CDF (just sort and draw)
// - Format:
//   [x] Text
//      - Each line is a record
//      - Each column is a field of the record
//   [x] Binary need offset / stride / type
// - Layout:
//   [x] No imgui.ini
//   [x] Automatic docking grid layout
//   [x] Grid position overrides for multiple plots
// - Style:
//   [x] Colors
//   [x] Title/Axis labels
//   [x] Line labels (custom or deduced from format / input)
//   [x] Axis endpoints
//   [x] Grid
//   [x] Imgui color themes
// - Platform:
//   [x] Support vulkan 1.0
//   [x] Optional vulkan debug / validation (controlled by flags)
//   [x] Window size controls
//   [ ] Vulkan device override (maybe rely on vulkan configurator / system specific options?)
// - Extra features:
//   [ ] Read stdin over time and append to plot
//   [ ] Passthrough (input is also written as stdout, useful for progress bars)
//   [ ] Ringbuffer / windowed plot (keep n last points, mostly useful for pipes / data coming over time)
//   [ ] Save as png (stb_image_write) and off-surface render
//   [ ] Data subsampling for drawing perf (could also be done as preprocess, but could be convenient)
//   [ ] add options for extra stats (mean, median, quartiles, stddev) -> extra lines in histogram?
// - Cleanup:
//   [ ] Move stuff to more files, move apps in own directory
//   [ ] Make XPG more standalone / clean

// Parsing logic:
// -t --text (<column>:<id>)*-> default, if not specified we could also autodetect by trying to parse as long as valid nums come out
// -b --binary <dtype>,offset[:<id>]
// -b 64 u32,0 f64,16
// ids: id specified assigned in left to right order with enum style continuation

// Plotting sources:

// One liners:
// - Plot text data from stdin (one line per column)
//   plot -
// - Explicit: Plot text data from stdin (one line per column)
//   plot - -p :0- m=o,s,d s=4 w=4 t=title x=xlabel y=ylable g=false c=red,green,blue b=0 t=1000 l=0 r=1000

// Plot (-p):
//  - style:
//  m | marker <list[char]>
//  s | size   <int>
//  w | width  <int>
//  c | color  <list[string]>
//  g | grid   <bool> = true
//  - labels:
//  T | title  <string>
//  X | xaxis label <string>
//  Y | yaxis label <string>
//  - limits:
//  t | top    <scalar>
//  b | bottom <scalar>
//  l | left   <scalar>
//  r | right  <scalar>
//  - scale
//  x <string>
//  y <string>
//  - layout:
//  q | quadrant <scalar, scalar>


// - Plot text data from stdin (x, y)
//   plot - -p 0:1
// - Plot text data from stdin (x, y1, y2, y3)
//   plot - -p 0:1-3
// - Plot text data from stdin (x, y1, y2, y3, ...)
//   plot - -p 0:1-

// - Scatter plot
//   plot - --style m=<marker>[,size] l=<line>[,width]

// Notes:
// - For formatting and layout it would be nice to have a very simple cmdline syntax to specify where inputs come from and where outputs go
//      - Needs to be intuitive and easy to remember -> simplest case must be minimal
//      - would be nice if we have a syntax for repeat/use until last (kind of like a small regex)


ImPlotMarker CharToMarker(char c) {
    switch(c) {
        case 'o': return ImPlotMarker_Circle;    // a circle marker (default)
        case 's': return ImPlotMarker_Square;    // a square maker
        case 'd': return ImPlotMarker_Diamond;   // a diamond marker
        case '^': return ImPlotMarker_Up;        // an upward-pointing triangle marker
        case 'v': return ImPlotMarker_Down;      // an downward-pointing triangle marker
        case 'l': return ImPlotMarker_Left;      // an leftward-pointing triangle marker
        case 'r': return ImPlotMarker_Right;     // an rightward-pointing triangle marker
        case 'x': return ImPlotMarker_Cross;     // a cross marker (not fillable)
        case 'p': return ImPlotMarker_Plus;      // a plus marker (not fillable)
        case 'a': return ImPlotMarker_Asterisk;  // a asterisk marker (not fillable)
        case 'n': return ImPlotMarker_None;      // no marker
        default: return INT_MAX;
    }
}

ImPlotScale StringToScale(const std::string& s) {
    if(s == "symlog" || s == "logsym") {
        return ImPlotScale_SymLog;
    } else if(s == "log" || s == "log10") {
        return ImPlotScale_Log10;
    } else if(s == "lin" || s == "linear") {
        return ImPlotScale_Linear;
    } else {
        return INT_MAX;
    }
}

ImPlotBin StringToAutoBin(const std::string& s) {
    if(s == "sqrt") {
        return ImPlotBin_Sqrt;
    } else if(s == "sturges") {
        return ImPlotBin_Sturges;
    } else if(s == "rice") {
        return ImPlotBin_Rice;
    } else if(s == "scott") {
        return ImPlotBin_Scott;
    } else {
        return INT_MAX;
    }
}

bool SplitAtChar(const std::string& input, char c, std::string& a, std::string& b) {
    size_t i = input.find_first_of(c);
    if(i == std::string::npos) return false;

    a = input.substr(0, i);
    b = input.substr(i + 1);
    return true;
}

void Fatal(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    exit(1);
}


std::string Lower(const std::string s) {
    std::string l = s;
    for(auto& c: l) {
        c = std::tolower(c);
    }
    return l;
}

template<typename T>
T Parse(const std::string& s, const std::string& opt);

template<> bool Parse<bool>(const std::string& s, const std::string& opt) {
    if(s.size() == 0) {
        Fatal("Invalid empty value for boolean option %s\n", opt.c_str());
    }

    std::string l = Lower(s);
    if(l == "1" || l == "true" || l == "on") {
        return true;
    } else if(l == "0" || l == "false" || l == "off") {
        return false;
    } else {
        Fatal("Invalid value \"%s\" for boolean option %s\n", s.c_str(), opt.c_str());
    }

    return false;
}

template<> double Parse<double>(const std::string& s, const std::string& opt) {
    if(s.size() == 0) {
        Fatal("Invalid empty value for scalar option %s\n", opt.c_str());
    }

    try {
        char* next_it;
        double val = strtod(s.c_str(), &next_it);
        if(next_it == s.c_str() || next_it != s.c_str() + s.size()) {
            throw std::exception();
        }
        return val;
    } catch (...) {
        Fatal("Invalid value \"%s\" for scalar option %s\n", s.c_str(), opt.c_str());
    }

    return 0;
}

template<> std::string Parse<std::string>(const std::string& s, const std::string& opt) {
    if(s.size() == 0) {
        Fatal("Invalid empty value for string option %s\n", opt.c_str());
    }
    return s;

}

s32 StringToInt(const std::string& s) {
    char* next_it;
    s32 val = strtol(s.c_str(), &next_it, 10);
    if(next_it == s.c_str() || next_it != s.c_str() + s.size()) {
        throw std::exception();
    }
    return val;
}

BinaryType StringToBinaryType(const std::string& s) {
    std::string l = Lower(s);
    if     (l == "f32")  return BinaryType::f32;
    else if(l == "f64")  return BinaryType::f64;
    else if(l == "i8" )  return BinaryType::i8;
    else if(l == "i16")  return BinaryType::i16;
    else if(l == "i32")  return BinaryType::i32;
    else if(l == "i64")  return BinaryType::i64;
    else if(l == "u8" )  return BinaryType::u8;
    else if(l == "u16")  return BinaryType::u16;
    else if(l == "u32")  return BinaryType::u32;
    else if(l == "u64")  return BinaryType::u64;
    else if(l == "void") return BinaryType::Void;

    return BinaryType::None;
}

usize BinaryTypeSize(BinaryType type) {
    switch(type) {
        case BinaryType::f32: return 4;
        case BinaryType::f64: return 8;
        case BinaryType::i8 : return 1;
        case BinaryType::i16: return 2;
        case BinaryType::i32: return 4;
        case BinaryType::i64: return 8;
        case BinaryType::u8 : return 1;
        case BinaryType::u16: return 2;
        case BinaryType::u32: return 4;
        case BinaryType::u64: return 8;
        case BinaryType::Void: return 0;
        default: return 0;
    }
}

double StringToDouble(const std::string& s) {
    char* next_it;
    double val = strtod(s.c_str(), &next_it);
    if(next_it == s.c_str() || next_it != s.c_str() + s.size()) {
        throw std::exception();
    }
    return val;
}

template<> AxisLimits Parse<AxisLimits>(const std::string& s, const std::string& opt) {
    if(s.size() == 0) {
        Fatal("Invalid empty value for string option %s\n", opt.c_str());
    }

    std::string min, max;
    if(!SplitAtChar(s, '-', min, max)) {
        Fatal("Invalid value \"%s\" for option %s, must be a range in the format <min>-<max> (e.g. 10-100)\n", s.c_str(), opt.c_str());
    }

    double min_val, max_val;
    try {
        min_val = StringToDouble(min);
    } catch (...) {
        Fatal("Invalid value \"%s\" for min value of option %s\n", min.c_str(), opt.c_str());
    }
    try {
        max_val = StringToDouble(max);
    } catch (...) {
        Fatal("Invalid value \"%s\" for max value of option %s\n", max.c_str(), opt.c_str());
    }

    AxisLimits limits = {};
    limits.set = true;
    limits.min = min_val;
    limits.max = max_val;
    return limits;
}

template<> Color Parse<Color>(const std::string& s, const std::string& opt) {
    static const std::unordered_map<std::string, ImU32> color_map {
        {"snow", 0xfffafaff },
        {"ghostwhite", 0xfffff8f8 },
        {"whitesmoke", 0xfff5f5f5 },
        {"gainsboro", 0xffdcdcdc },
        {"floralwhite", 0xfff0faff },
        {"oldlace", 0xffe6f5fd },
        {"linen", 0xffe6f0fa },
        {"antiquewhite", 0xffd7ebfa },
        {"papayawhip", 0xffd5efff },
        {"blanchedalmond", 0xffcdebff },
        {"bisque", 0xffc4e4ff },
        {"peachpuff", 0xffb9daff },
        {"navajowhite", 0xffaddeff },
        {"moccasin", 0xffb5e4ff },
        {"cornsilk", 0xffdcf8ff },
        {"ivory", 0xfff0ffff },
        {"lemonchiffon", 0xffcdfaff },
        {"seashell", 0xffeef5ff },
        {"honeydew", 0xfff0fff0 },
        {"mintcream", 0xfffafff5 },
        {"azure", 0xfffffff0 },
        {"aliceblue", 0xfffff8f0 },
        {"lavender", 0xfffae6e6 },
        {"lavenderblush", 0xfff5f0ff },
        {"mistyrose", 0xffe1e4ff },
        {"white", 0xffffffff },
        {"black", 0xff000000 },
        {"darkslategray", 0xff4f4f2f },
        {"darkslategrey", 0xff4f4f2f },
        {"dimgray", 0xff696969 },
        {"dimgrey", 0xff696969 },
        {"slategray", 0xff908070 },
        {"slategrey", 0xff908070 },
        {"lightslategray", 0xff998877 },
        {"lightslategrey", 0xff998877 },
        {"gray", 0xffbebebe },
        {"grey", 0xffbebebe },
        {"lightgrey", 0xffd3d3d3 },
        {"lightgray", 0xffd3d3d3 },
        {"midnightblue", 0xff701919 },
        {"navy", 0xff800000 },
        {"navyblue", 0xff800000 },
        {"cornflowerblue", 0xffed9564 },
        {"darkslateblue", 0xff8b3d48 },
        {"slateblue", 0xffcd5a6a },
        {"mediumslateblue", 0xffee687b },
        {"lightslateblue", 0xffff7084 },
        {"mediumblue", 0xffcd0000 },
        {"royalblue", 0xffe16941 },
        {"blue", 0xffff0000 },
        {"dodgerblue", 0xffff901e },
        {"deepskyblue", 0xffffbf00 },
        {"skyblue", 0xffebce87 },
        {"lightskyblue", 0xffface87 },
        {"steelblue", 0xffb48246 },
        {"lightsteelblue", 0xffdec4b0 },
        {"lightblue", 0xffe6d8ad },
        {"powderblue", 0xffe6e0b0 },
        {"paleturquoise", 0xffeeeeaf },
        {"darkturquoise", 0xffd1ce00 },
        {"mediumturquoise", 0xffccd148 },
        {"turquoise", 0xffd0e040 },
        {"cyan", 0xffffff00 },
        {"lightcyan", 0xffffffe0 },
        {"cadetblue", 0xffa09e5f },
        {"mediumaquamarine", 0xffaacd66 },
        {"aquamarine", 0xffd4ff7f },
        {"darkgreen", 0xff006400 },
        {"darkolivegreen", 0xff2f6b55 },
        {"darkseagreen", 0xff8fbc8f },
        {"seagreen", 0xff578b2e },
        {"mediumseagreen", 0xff71b33c },
        {"lightseagreen", 0xffaab220 },
        {"palegreen", 0xff98fb98 },
        {"springgreen", 0xff7fff00 },
        {"lawngreen", 0xff00fc7c },
        {"green", 0xff00ff00 },
        {"chartreuse", 0xff00ff7f },
        {"mediumspringgreen", 0xff9afa00 },
        {"greenyellow", 0xff2fffad },
        {"limegreen", 0xff32cd32 },
        {"yellowgreen", 0xff32cd9a },
        {"forestgreen", 0xff228b22 },
        {"olivedrab", 0xff238e6b },
        {"darkkhaki", 0xff6bb7bd },
        {"khaki", 0xff8ce6f0 },
        {"palegoldenrod", 0xffaae8ee },
        {"lightgoldenrodyellow", 0xffd2fafa },
        {"lightyellow", 0xffe0ffff },
        {"yellow", 0xff00ffff },
        {"gold", 0xff00d7ff },
        {"lightgoldenrod", 0xff82ddee },
        {"goldenrod", 0xff20a5da },
        {"darkgoldenrod", 0xff0b86b8 },
        {"rosybrown", 0xff8f8fbc },
        {"indianred", 0xff5c5ccd },
        {"saddlebrown", 0xff13458b },
        {"sienna", 0xff2d52a0 },
        {"peru", 0xff3f85cd },
        {"burlywood", 0xff87b8de },
        {"beige", 0xffdcf5f5 },
        {"wheat", 0xffb3def5 },
        {"sandybrown", 0xff60a4f4 },
        {"tan", 0xff8cb4d2 },
        {"chocolate", 0xff1e69d2 },
        {"firebrick", 0xff2222b2 },
        {"brown", 0xff2a2aa5 },
        {"darksalmon", 0xff7a96e9 },
        {"salmon", 0xff7280fa },
        {"lightsalmon", 0xff7aa0ff },
        {"orange", 0xff00a5ff },
        {"darkorange", 0xff008cff },
        {"coral", 0xff507fff },
        {"lightcoral", 0xff8080f0 },
        {"tomato", 0xff4763ff },
        {"orangered", 0xff0045ff },
        {"red", 0xff0000ff },
        {"hotpink", 0xffb469ff },
        {"deeppink", 0xff9314ff },
        {"pink", 0xffcbc0ff },
        {"lightpink", 0xffc1b6ff },
        {"palevioletred", 0xff9370db },
        {"maroon", 0xff6030b0 },
        {"mediumvioletred", 0xff8515c7 },
        {"violetred", 0xff9020d0 },
        {"magenta", 0xffff00ff },
        {"violet", 0xffee82ee },
        {"plum", 0xffdda0dd },
        {"orchid", 0xffd670da },
        {"mediumorchid", 0xffd355ba },
        {"darkorchid", 0xffcc3299 },
        {"darkviolet", 0xffd30094 },
        {"blueviolet", 0xffe22b8a },
        {"purple", 0xfff020a0 },
        {"mediumpurple", 0xffdb7093 },
        {"thistle", 0xffd8bfd8 },
        {"darkgrey", 0xffa9a9a9 },
        {"darkgray", 0xffa9a9a9 },
        {"darkblue", 0xff8b0000 },
        {"darkcyan", 0xff8b8b00 },
        {"darkmagenta", 0xff8b008b },
        {"darkred", 0xff00008b },
        {"lightgreen", 0xff90ee90 },
    };

    if(s.size() == 0) {
        Fatal("Invalid empty value for color option %s\n", opt.c_str());
    }

    if(s[0] == '#') {
        // TODO: parse hex color
    }

    auto it = color_map.find(s);
    if(it != color_map.end()) {
        Color c = {};
        c.color = it->second;
        return c;
    }

    Fatal("Invalid value \"%s\" for color option %s\n", s.c_str(), opt.c_str());
    return {};
}

template<> ImPlotMarker_ Parse<ImPlotMarker_>(const std::string& s, const std::string& opt) {
    if(s.size() == 0) {
        Fatal("Invalid empty value for marker option %s\n", opt.c_str());
    }
    if(s.size() != 1) {
        Fatal("Invalid value \"%s\" for marker option %s must be a single character\n", s.c_str(), opt.c_str());
    }

    int marker = CharToMarker(s[0]);
    if(marker == INT_MAX) {
        Fatal("Invalid value \"%s\" for marker option %s must be one of o,s,d,^,v,l,r,x,p,a,n \n", s.c_str(), opt.c_str());
    }

    return (ImPlotMarker_)marker;
}

template<> ImPlotScale_ Parse<ImPlotScale_>(const std::string& s, const std::string& opt) {
    if(s.size() == 0) {
        Fatal("Invalid empty value for scale option %s\n", opt.c_str());
    }

    std::string l = Lower(s);
    int scale = StringToScale(l);
    if(scale == INT_MAX) {
        Fatal("Invalid value \"%s\" for scale option %s must be one of linear,log10,symlog \n", s.c_str(), opt.c_str());
    }

    return (ImPlotScale_)scale;
}

template<> ImPlotBin_ Parse<ImPlotBin_>(const std::string& s, const std::string& opt) {
    if(s.size() == 0) {
        Fatal("Invalid empty value for binning option %s\n", opt.c_str());
    }

    int bins = 0;

    try {
        bins = StringToInt(s);
        return (ImPlotBin_)bins;
    } catch (...) {

    }

    std::string l = Lower(s);
    bins = StringToAutoBin(l);
    if(bins == INT_MAX) {
        Fatal("Invalid value \"%s\" for binning option %s must be a positive integer or sturges,sqrt,rice,scott \n", s.c_str(), opt.c_str());
    }

    return (ImPlotBin_)bins;
}

template<typename T>
std::vector<T> ParseList(const std::string& s, const std::string& opt) {
    if(s.size() == 0) {
        Fatal("Unexpected empty list for option %s\n", opt.c_str());
    }

    std::vector<T> values;
    size_t it = 0;
    while(it < s.size()) {
        size_t pos = s.find_first_of(',', it);

        T v = Parse<T>(s.substr(it, pos == std::string::npos ? pos : pos - it), opt);
        values.push_back(v);

        if(pos == std::string::npos) {
            break;
        }
        it = pos + 1;
    }

    return values;
}

void SetDarkTheme()
{
    ImGuiStyle* style = &ImGui::GetStyle();
    ImVec4* colors = style->Colors;

    using namespace ImGui;
    using namespace Spectrum::Dark;
    colors[ImGuiCol_Text] = ColorConvertU32ToFloat4(GRAY800); // text on hovered controls is gray900
    colors[ImGuiCol_TextDisabled] = ColorConvertU32ToFloat4(GRAY500);
    colors[ImGuiCol_WindowBg] = ColorConvertU32ToFloat4(GRAY100);
    colors[ImGuiCol_ChildBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg] = ColorConvertU32ToFloat4(GRAY50); // not sure about this. Note: applies to tooltips too.
    colors[ImGuiCol_Border] = ColorConvertU32ToFloat4(GRAY300);
    colors[ImGuiCol_BorderShadow] = ColorConvertU32ToFloat4(NONE); // We don't want shadows. Ever.
    colors[ImGuiCol_FrameBg] = ColorConvertU32ToFloat4(GRAY75); // this isnt right, spectrum does not do this, but it's a good fallback
    colors[ImGuiCol_FrameBgHovered] = ColorConvertU32ToFloat4(GRAY50);
    colors[ImGuiCol_FrameBgActive] = ColorConvertU32ToFloat4(GRAY200);
    colors[ImGuiCol_TitleBg] = ColorConvertU32ToFloat4(GRAY300); // those titlebar values are totally made up, spectrum does not have this.
    colors[ImGuiCol_TitleBgActive] = ColorConvertU32ToFloat4(GRAY200);
    colors[ImGuiCol_TitleBgCollapsed] = ColorConvertU32ToFloat4(GRAY400);
    colors[ImGuiCol_MenuBarBg] = ColorConvertU32ToFloat4(GRAY100);
    colors[ImGuiCol_ScrollbarBg] = ColorConvertU32ToFloat4(GRAY100); // same as regular background
    colors[ImGuiCol_ScrollbarGrab] = ColorConvertU32ToFloat4(GRAY400);
    colors[ImGuiCol_ScrollbarGrabHovered] = ColorConvertU32ToFloat4(GRAY600);
    colors[ImGuiCol_ScrollbarGrabActive] = ColorConvertU32ToFloat4(GRAY700);
    colors[ImGuiCol_CheckMark] = ColorConvertU32ToFloat4(BLUE500);
    colors[ImGuiCol_SliderGrab] = ColorConvertU32ToFloat4(GRAY700);
    colors[ImGuiCol_SliderGrabActive] = ColorConvertU32ToFloat4(GRAY800);
    colors[ImGuiCol_Button] = ColorConvertU32ToFloat4(GRAY75); // match default button to Spectrum's 'Action Button'.
    colors[ImGuiCol_ButtonHovered] = ColorConvertU32ToFloat4(GRAY50);
    colors[ImGuiCol_ButtonActive] = ColorConvertU32ToFloat4(GRAY200);
    colors[ImGuiCol_Header] = ColorConvertU32ToFloat4(BLUE400);
    colors[ImGuiCol_HeaderHovered] = ColorConvertU32ToFloat4(BLUE500);
    colors[ImGuiCol_HeaderActive] = ColorConvertU32ToFloat4(BLUE600);
    colors[ImGuiCol_Separator] = ColorConvertU32ToFloat4(GRAY400);
    colors[ImGuiCol_SeparatorHovered] = ColorConvertU32ToFloat4(GRAY600);
    colors[ImGuiCol_SeparatorActive] = ColorConvertU32ToFloat4(GRAY700);
    colors[ImGuiCol_ResizeGrip] = ColorConvertU32ToFloat4(GRAY400);
    colors[ImGuiCol_ResizeGripHovered] = ColorConvertU32ToFloat4(GRAY600);
    colors[ImGuiCol_ResizeGripActive] = ColorConvertU32ToFloat4(GRAY700);
    colors[ImGuiCol_PlotLines] = ColorConvertU32ToFloat4(BLUE400);
    colors[ImGuiCol_PlotLinesHovered] = ColorConvertU32ToFloat4(BLUE600);
    colors[ImGuiCol_PlotHistogram] = ColorConvertU32ToFloat4(BLUE400);
    colors[ImGuiCol_PlotHistogramHovered] = ColorConvertU32ToFloat4(BLUE600);
    colors[ImGuiCol_TextSelectedBg] = ColorConvertU32ToFloat4((BLUE400 & 0x00FFFFFF) | 0x33000000);
    colors[ImGuiCol_DragDropTarget] = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
    colors[ImGuiCol_NavHighlight] = ColorConvertU32ToFloat4((GRAY900 & 0x00FFFFFF) | 0x0A000000);
    colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.20f, 0.20f, 0.20f, 0.35f);
}

void SetLightTheme()
{
    ImGuiStyle* style = &ImGui::GetStyle();
    ImVec4* colors = style->Colors;

    using namespace ImGui;
    using namespace Spectrum::Light;
    colors[ImGuiCol_Text] = ColorConvertU32ToFloat4(GRAY800); // text on hovered controls is gray900
    colors[ImGuiCol_TextDisabled] = ColorConvertU32ToFloat4(GRAY500);
    colors[ImGuiCol_WindowBg] = ColorConvertU32ToFloat4(GRAY100);
    colors[ImGuiCol_ChildBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg] = ColorConvertU32ToFloat4(GRAY50); // not sure about this. Note: applies to tooltips too.
    colors[ImGuiCol_Border] = ColorConvertU32ToFloat4(GRAY300);
    colors[ImGuiCol_BorderShadow] = ColorConvertU32ToFloat4(NONE); // We don't want shadows. Ever.
    colors[ImGuiCol_FrameBg] = ColorConvertU32ToFloat4(GRAY75); // this isnt right, spectrum does not do this, but it's a good fallback
    colors[ImGuiCol_FrameBgHovered] = ColorConvertU32ToFloat4(GRAY50);
    colors[ImGuiCol_FrameBgActive] = ColorConvertU32ToFloat4(GRAY200);
    colors[ImGuiCol_TitleBg] = ColorConvertU32ToFloat4(GRAY300); // those titlebar values are totally made up, spectrum does not have this.
    colors[ImGuiCol_TitleBgActive] = ColorConvertU32ToFloat4(GRAY200);
    colors[ImGuiCol_TitleBgCollapsed] = ColorConvertU32ToFloat4(GRAY400);
    colors[ImGuiCol_MenuBarBg] = ColorConvertU32ToFloat4(GRAY100);
    colors[ImGuiCol_ScrollbarBg] = ColorConvertU32ToFloat4(GRAY100); // same as regular background
    colors[ImGuiCol_ScrollbarGrab] = ColorConvertU32ToFloat4(GRAY400);
    colors[ImGuiCol_ScrollbarGrabHovered] = ColorConvertU32ToFloat4(GRAY600);
    colors[ImGuiCol_ScrollbarGrabActive] = ColorConvertU32ToFloat4(GRAY700);
    colors[ImGuiCol_CheckMark] = ColorConvertU32ToFloat4(BLUE500);
    colors[ImGuiCol_SliderGrab] = ColorConvertU32ToFloat4(GRAY700);
    colors[ImGuiCol_SliderGrabActive] = ColorConvertU32ToFloat4(GRAY800);
    colors[ImGuiCol_Button] = ColorConvertU32ToFloat4(GRAY75); // match default button to Spectrum's 'Action Button'.
    colors[ImGuiCol_ButtonHovered] = ColorConvertU32ToFloat4(GRAY50);
    colors[ImGuiCol_ButtonActive] = ColorConvertU32ToFloat4(GRAY200);
    colors[ImGuiCol_Header] = ColorConvertU32ToFloat4(BLUE400);
    colors[ImGuiCol_HeaderHovered] = ColorConvertU32ToFloat4(BLUE500);
    colors[ImGuiCol_HeaderActive] = ColorConvertU32ToFloat4(BLUE600);
    colors[ImGuiCol_Separator] = ColorConvertU32ToFloat4(GRAY400);
    colors[ImGuiCol_SeparatorHovered] = ColorConvertU32ToFloat4(GRAY600);
    colors[ImGuiCol_SeparatorActive] = ColorConvertU32ToFloat4(GRAY700);
    colors[ImGuiCol_ResizeGrip] = ColorConvertU32ToFloat4(GRAY400);
    colors[ImGuiCol_ResizeGripHovered] = ColorConvertU32ToFloat4(GRAY600);
    colors[ImGuiCol_ResizeGripActive] = ColorConvertU32ToFloat4(GRAY700);
    colors[ImGuiCol_PlotLines] = ColorConvertU32ToFloat4(BLUE400);
    colors[ImGuiCol_PlotLinesHovered] = ColorConvertU32ToFloat4(BLUE600);
    colors[ImGuiCol_PlotHistogram] = ColorConvertU32ToFloat4(BLUE400);
    colors[ImGuiCol_PlotHistogramHovered] = ColorConvertU32ToFloat4(BLUE600);
    colors[ImGuiCol_TextSelectedBg] = ColorConvertU32ToFloat4((BLUE400 & 0x00FFFFFF) | 0x33000000);
    colors[ImGuiCol_DragDropTarget] = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
    colors[ImGuiCol_NavHighlight] = ColorConvertU32ToFloat4((GRAY900 & 0x00FFFFFF) | 0x0A000000);
    colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.20f, 0.20f, 0.20f, 0.35f);
}

std::vector<Range> ParseRanges(const std::string b) {
    std::vector<Range> ranges;
    if(!b.empty()) {
        size_t it = 0;
        while(it < b.size()) {
            Range range = {};

            size_t pos = b.find_first_of(',', it);

            std::string r = b.substr(it, pos == std::string::npos ? pos : pos - it);
            std::string begin, end;
            if(SplitAtChar(r, '-', begin, end)) {
                if(begin.empty()) {
                    range.begin = 0;
                } else {
                    try {
                        range.begin = StringToInt(begin);
                    } catch(...) {
                        Fatal("Plot y range begin must be an integer. Found \"%s\"\n", begin.c_str());
                    }
                }
                if(end.empty()) {
                    range.end = INT_MAX;
                } else {
                    try {
                        range.end = StringToInt(end);
                    } catch(...) {
                        Fatal("Plot y range end must be an integer. Found \"%s\"\n", end.c_str());
                    }
                }
            } else {
                try {
                    s32 v = StringToInt(r);
                    range.begin = v;
                    range.end = v;
                } catch(...) {
                    Fatal("Plot y must be an integer or a range. Found \"%s\"\n", r.c_str());
                }
            }

            ranges.push_back(range);

            if(pos == std::string::npos) {
                break;
            }
            it = pos + 1;
        }
    } else {
        Range range = {};
        range.begin = 0;
        range.end = INT_MAX;

        ranges.push_back(range);
    }

    return ranges;
}
Lines ParseLineRange(const std::string& a, const std::string& b) {
    // a -> optional number
    // b -> comma separated list of ranges
    // range -> optional number - optional number
    Lines lines = {};
    if(!a.empty()) {
        try {
            lines.x = StringToInt(a);
        } catch(...) {
            Fatal("Plot x must be an integer. Found \"%s\"\n", a.c_str());
        }
    } else {
        lines.x = -1;
    }

    lines.ys = ParseRanges(b);

    return lines;
}

int main(int argc, char** argv) {
    CLI::App args{"Command line utility for plotting"};

    // Define options
    std::string input;
    args.add_option("input", input, "Input file or empty for stdin");

    std::vector<std::string> binary;
    CLI::Option* opt_binary = args.add_option("-b,--binary", binary, "Binary format");

    std::vector<std::vector<std::string>> plots_options;
    args.add_option("-p,--plot", plots_options, "Plot, followed by input range and options")->take_all()->expected(0,-1);

    std::vector<std::vector<std::string>> histogram_options;
    args.add_option("-t,--hist", histogram_options, "Histogram, followed by input range and options")->take_all()->expected(0,-1);

    int window_width = 1600;
    args.add_option("-W,--width", window_width, "Window width")->check(CLI::Range(1, 100000));

    int window_height = 900;
    args.add_option("-H,--height", window_height, "Window height")->check(CLI::Range(1, 100000));

    float font_size = 16.0f;
    args.add_option("-F,--font-size", font_size, "Font size")->check(CLI::Range(1.0f, 100.0f));

    bool light_theme = false;
    args.add_flag("-L,--light", light_theme, "Set light theme");

    bool vulkan_verbose = false;
    args.add_flag("--vulkan-verbose", vulkan_verbose, "Print vulkan information");

    bool enable_vulkan_validation = false;
    args.add_flag("--vulkan-validation", enable_vulkan_validation, "Enable vulkan validation layer, if available");

    CLI11_PARSE(args, argc, argv);

    Data data = {};
    // Parse all data upfront
    if(opt_binary->count()) {
        // Parse binary input
        Array<BinaryFormat> formats;
        u64 max_size = 0;

        u64 offset = 0;
        for(auto& f: binary) {
            std::string a, b;

            BinaryFormat format = {};

            std::string& e_type = f;
            format.offset = offset;

            if(SplitAtChar(f, ':', a, b)) {
                e_type = a;
                if(b.size() == 0) {
                    Fatal("Invalid empty offset in binary format string\n");
                }

                if(b[0] == '+') {
                    int off;
                    try {
                        off = StringToInt(b.substr(1));
                    } catch(...) {
                        Fatal("Invalid offset \"%s\" in binary format string\n", b.c_str());
                    }
                    format.offset += off;
                } else {
                    int off;
                    try {
                        off = StringToInt(b);
                    } catch(...) {
                        Fatal("Invalid offset \"%s\" in binary format string\n", b.c_str());
                    }
                    format.offset = off;
                }
            }

            if(e_type.size() == 0) {
                Fatal("Invalid empty type in binary format string\n");
            }

            format.type = StringToBinaryType(e_type);
            format.size = BinaryTypeSize(format.type);
            if(format.type == BinaryType::None) {
                Fatal("Invalid binary type \"%s\" in binary format. Must be one of i8,i16,i32,i64,u8,u16,u32,u64,f32,f64,void\n", e_type.c_str());
            }

            offset = format.offset + format.size;
            max_size = std::max(max_size, offset);

            if(format.type != BinaryType::Void) {
                formats.add(format);
            }
        }

        if(max_size >= 1 << 30) {
            Fatal("Binary record size must be < 1GB. Got %llu bytes\n", (unsigned long long)max_size);
        }

        FILE* file;
        if(input == "") {
            freopen(NULL, "rb", stdin);
            file = stdin;
        } else {
            file = fopen(input.c_str(), "rb");
        }

        if(!file || ferror(file)) {
            Fatal("Could not read %s: %s (os error %d)\n", input.c_str(), strerror(errno), errno);
        }

        Array<u8> row(max_size);
        while(!feof(file) && !ferror(file)) {
            size_t count_read = fread(row.data, row.length, 1, file);

            // Read failed
            if(count_read != 1) {
                if(ferror(file)) {
                    Fatal("Failed to read a record of size %u. Input file is likely not a multiple of specified record size.\n", (unsigned)row.length);
                }
                if(feof(file)) {
                    break;
                }
                continue;
            }

            for(usize i = 0; i < formats.length; i++) {
                BinaryFormat& fmt = formats[i];
                void* ptr = row.data + fmt.offset;
                assert(fmt.offset + fmt.size <= row.length);

                double value = 0.0;
                switch(fmt.type) {
                    case BinaryType::f32: value = (double)*(f32*)ptr; break;
                    case BinaryType::f64: value = (double)*(f64*)ptr; break;
                    case BinaryType::i8 : value = (double)*(s8 *)ptr; break;
                    case BinaryType::i16: value = (double)*(s16*)ptr; break;
                    case BinaryType::i32: value = (double)*(s32*)ptr; break;
                    case BinaryType::i64: value = (double)*(s64*)ptr; break;
                    case BinaryType::u8 : value = (double)*(u8 *)ptr; break;
                    case BinaryType::u16: value = (double)*(u16*)ptr; break;
                    case BinaryType::u32: value = (double)*(u32*)ptr; break;
                    case BinaryType::u64: value = (double)*(u64*)ptr; break;
                    default: assert(false);
                }

                data.data.add(value);
            }
        }

        if(input != "") {
            fclose(file);
        }

        data.columns = formats.length;
    } else {
        // Parse text input
        std::string line;

        std::istream* stream;
        if(input == "") {
            stream = &std::cin;
        } else {
            stream = new std::ifstream (input);
        }

        if(stream->fail()) {
            Fatal("Could not read %s: %s (os error %d)\n", input.c_str(), strerror(errno), errno);
        }

        s64 expected_count = -1;
        s64 line_number = 0;
        while(!stream->eof() && !stream->fail()) {
            std::getline(*stream, line);
            line_number++;

            const char* end = line.c_str() + line.size();
            const char* it = line.c_str();

            s64 count = 0;
            while(it != end) {
                char* next_it;
                double val = 0.0;
                try {
                    val = strtod(it, &next_it);
                    if(next_it == it) {
                        throw std::exception();
                    }
                    it = next_it;
                    count += 1;

                    data.data.add(val);
                } catch (...) {
                    Fatal("%s(%d): Failed to parse value \"%s\"\n", input.c_str(), line_number,  it);
                }
                // Skip whitespace
                while(it != end && std::isspace(*it)) it++;
            }

            if(count == 0) continue;

            if(expected_count == -1) {
                expected_count = count;
            } else if(expected_count != count) {
                Fatal("%s(%d): Expected %d columns, found %d\n", input.c_str(), line_number, expected_count, count);
            }
        }
        if(expected_count <= 0) {
            Fatal("Empty input\n");
        }
        if(input != "") {
            delete stream;
        }
        data.columns = expected_count;
    }

    std::vector<Plot> plots;

    // Parse plots
    for(auto& p: plots_options) {
        Plot plot = {};
        for(auto& o: p) {
            std::string a, b;
            if(SplitAtChar(o, '=', a, b)) {
                // Parse option
                if       (a == "m" || a == "marker") {
                    plot.markers = ParseList<ImPlotMarker_>(b, a);
                } else if(a == "s" || a == "size") {
                    plot.size = Parse<double>(b, a);
                } else if(a == "w" || a == "width") {
                    plot.width = Parse<double>(b, a);
                } else if(a == "c" || a == "color") {
                    plot.colors = ParseList<Color>(b, a);
                } else if(a == "g" || a == "grid") {
                    plot.grid = Parse<bool>(b, a);
                } else if(a == "T" || a == "title") {
                    plot.title = Parse<std::string>(b, a);
                } else if(a == "X" || a == "xlabel") {
                    plot.xlabel = Parse<std::string>(b, a);
                } else if(a == "Y" || a == "ylabel") {
                    plot.ylabel = Parse<std::string>(b, a);
                } else if(a == "n" || a == "name") {
                    plot.names = ParseList<std::string>(b, a);
                } else if(a == "xlim" || a == "xlimit" || a == "xlimits") {
                    plot.xlimits = Parse<AxisLimits>(b, a);
                } else if(a == "ylim" || a == "ylimit" || a == "ylimits") {
                    plot.ylimits = Parse<AxisLimits>(b, a);
                } else if(a == "xscale") {
                    plot.xscale = Parse<ImPlotScale_>(b, a);
                } else if(a == "yscale") {
                    plot.yscale = Parse<ImPlotScale_>(b, a);
                } else {
                    Fatal("Unknown option %s\n", a.c_str());
                }
            } else if(SplitAtChar(o, ':', a, b)) {
                Lines lines = ParseLineRange(a, b);
                plot.lines.push_back(lines);
            } else {
                Fatal("Plot argument must be range (e.g. \"0:1\") or option (e.g. \"k=v\")\n", a.c_str());
            }
        }

        // No line specified default to all y axis.
        if(plot.lines.empty()) {
            Lines lines = {};
            lines.x = -1;

            Range range = {};
            range.begin = 0;
            range.end = INT_MAX;
            lines.ys.push_back(range);

            plot.lines.push_back(lines);
        }

        plots.push_back(plot);
    }

    // Parse histograms
    for(auto& h: histogram_options) {
        Plot plot = {};
        plot.kind = PlotKind::Histogram;
        for(auto& o: h) {
            std::string a, b;
            if(SplitAtChar(o, '=', a, b)) {
                // Parse option
                if       (a == "m" || a == "marker") {
                    plot.markers = ParseList<ImPlotMarker_>(b, a);
                } else if(a == "s" || a == "size") {
                    plot.size = Parse<double>(b, a);
                } else if(a == "w" || a == "width") {
                    plot.width = Parse<double>(b, a);
                } else if(a == "c" || a == "color") {
                    plot.colors = ParseList<Color>(b, a);
                } else if(a == "g" || a == "grid") {
                    plot.grid = Parse<bool>(b, a);
                } else if(a == "T" || a == "title") {
                    plot.title = Parse<std::string>(b, a);
                } else if(a == "X" || a == "xlabel") {
                    plot.xlabel = Parse<std::string>(b, a);
                } else if(a == "Y" || a == "ylabel") {
                    plot.ylabel = Parse<std::string>(b, a);
                } else if(a == "n" || a == "name") {
                    plot.names = ParseList<std::string>(b, a);
                } else if(a == "xlim" || a == "xlimit" || a == "xlimits") {
                    plot.xlimits = Parse<AxisLimits>(b, a);
                } else if(a == "ylim" || a == "ylimit" || a == "ylimits") {
                    plot.ylimits = Parse<AxisLimits>(b, a);
                } else if(a == "xscale") {
                    plot.xscale = Parse<ImPlotScale_>(b, a);
                } else if(a == "yscale") {
                    plot.yscale = Parse<ImPlotScale_>(b, a);
                } else if(a == "b" || a == "bin" || a == "binning" || a == "bins") {
                    plot.histogram_bins = Parse<ImPlotBin_>(b, a);
                } else if(a == "h" || a == "hor" || a == "horizontal") {
                    if(Parse<bool>(b, a)) {
                        plot.histogram_flags |= ImPlotHistogramFlags_Horizontal;
                    }
                } else if(a == "cum" || a == "cdf" || a == "cumulative") {
                    if(Parse<bool>(b, a)) {
                        plot.histogram_flags |= ImPlotHistogramFlags_Cumulative;
                    }
                } else if(a == "norm" || a == "normal" || a == "normalize") {
                    if(Parse<bool>(b, a)) {
                        plot.histogram_flags |= ImPlotHistogramFlags_Density;
                    }
                } else if(a == "r" || a == "range") {
                    plot.histogram_range = Parse<AxisLimits>(b, a);
                } else {
                    Fatal("Unknown option %s\n", a.c_str());
                }
            } else {
                Lines lines = {};
                lines.x = -1;
                lines.ys = ParseRanges(o);
                plot.lines.push_back(lines);
            }
        }

        // No line specified default to all y axis.
        if(plot.lines.empty()) {
            Lines lines = {};
            lines.x = -1;

            Range range = {};
            range.begin = 0;
            range.end = INT_MAX;
            lines.ys.push_back(range);

            plot.lines.push_back(lines);
        }

        plots.push_back(plot);
    }


    // Push a default plot if empty
    if(plots.empty()) {
        Plot plot = {};

        Lines lines = {};
        lines.x = -1;

        Range r = {};
        r.begin = 0;
        r.end = INT_MAX;
        lines.ys.push_back(r);

        plot.lines.push_back(lines);

        plots.push_back(plot);
    }

    // Initialize glfw.
    glfwInit();

    // Check if device supports vulkan.
    if (!glfwVulkanSupported()) {
        printf("Vulkan not found!\n");
        exit(1);
    }

    // Get instance extensions required by GLFW.
    u32 glfw_instance_extensions_count;
    const char** glfw_instance_extensions = glfwGetRequiredInstanceExtensions(&glfw_instance_extensions_count);

    // Vulkan initialization.
    Array<const char*> instance_extensions;
    for(u32 i = 0; i < glfw_instance_extensions_count; i++) {
        instance_extensions.add(glfw_instance_extensions[i]);
    }
    instance_extensions.add("VK_EXT_debug_report");

    Array<const char*> device_extensions;
    device_extensions.add("VK_KHR_swapchain");

    u32 vulkan_api_version = VK_API_VERSION_1_0;

    VulkanContext vk = {};
    if (InitializeVulkan(&vk, vulkan_api_version, instance_extensions, device_extensions,
            /*require_presentation_support =*/ true,
            /*dynamic_rendering =*/ false,
            /*enable_validation_layer =*/ enable_vulkan_validation,
            /*verbose =*/ vulkan_verbose
        ) != VulkanResult::SUCCESS) {
        printf("Failed to initialize vulkan\n");
        exit(1);
    }

    // Check if queue family supports image presentation.
    if (!glfwGetPhysicalDevicePresentationSupport(vk.instance, vk.physical_device, vk.queue_family_index)) {
        printf("Device does not support image presentation\n");
        exit(1);
    }

    VulkanWindow window = {};
    if (CreateVulkanWindow(&window, vk, "XPG", window_width, window_height, vulkan_verbose) != VulkanResult::SUCCESS) {
        printf("Failed to create vulkan window\n");
        return 1;
    }

#if 0
#ifdef _WIN32
    // Redraw during move / resize
    HWND hwnd = glfwGetWin32Window(window.window);
    HANDLE thread = CreateThread(0, 0, thread_proc, hwnd, 0, 0);
    if (thread) {
        CloseHandle(thread);
    }
#endif
#endif

    // Initialize queue and command allocator.
    VkResult result;

    VkQueue queue;
    vkGetDeviceQueue(vk.device, vk.queue_family_index, 0, &queue);

    Array<VulkanFrame> frames(window.images.length);
    Array<VkFramebuffer> framebuffers(window.images.length);

    for (usize i = 0; i < frames.length; i++) {
        VulkanFrame& frame = frames[i];

        VkCommandPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;// | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = vk.queue_family_index;

        result = vkCreateCommandPool(vk.device, &pool_info, 0, &frame.command_pool);
        assert(result == VK_SUCCESS);

        VkCommandBufferAllocateInfo allocate_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        allocate_info.commandPool = frame.command_pool;
        allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate_info.commandBufferCount = 1;

        result = vkAllocateCommandBuffers(vk.device, &allocate_info, &frame.command_buffer);
        assert(result == VK_SUCCESS);

        VkFenceCreateInfo fence_info = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(vk.device, &fence_info, 0, &frame.fence);

        CreateVulkanSemaphore(vk.device, &frame.acquire_semaphore);
        CreateVulkanSemaphore(vk.device, &frame.release_semaphore);

        framebuffers[i] = VK_NULL_HANDLE;
    }

    // Create descriptor pool for imgui.
    VkDescriptorPoolSize imgui_pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
    };

    VkDescriptorPoolCreateInfo imgui_descriptor_pool_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    imgui_descriptor_pool_info.flags = 0;
    imgui_descriptor_pool_info.maxSets = 1;
    imgui_descriptor_pool_info.pPoolSizes = imgui_pool_sizes;
    imgui_descriptor_pool_info.poolSizeCount = ArrayCount(imgui_pool_sizes);

    VkDescriptorPool imgui_descriptor_pool = 0;
    vkCreateDescriptorPool(vk.device, &imgui_descriptor_pool_info, 0, &imgui_descriptor_pool);

    // Setup window callbacks
    glfwSetWindowSizeCallback(window.window, Callback_WindowResize);
    glfwSetWindowRefreshCallback(window.window, Callback_WindowRefresh);
    glfwSetKeyCallback(window.window, Callback_Key);

    // Initialize ImGui.
    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.LogFilename = NULL;
    io.IniFilename = NULL;

    ImFontConfig font_cfg;
    font_cfg.FontDataOwnedByAtlas = false;
    io.Fonts->AddFontFromMemoryTTF(Roboto_Medium_ttf, Roboto_Medium_ttf_len, font_size, &font_cfg);

    if(light_theme) {
        SetLightTheme();
    } else {
        SetDarkTheme();
    }

    if (!ImGui_ImplGlfw_InitForVulkan(window.window, true)) {
        printf("Failed to initialize ImGui\n");
        exit(1);
    }

    // TODO: MSAA
    ImGui_ImplVulkan_InitInfo imgui_vk_init_info = {};
    imgui_vk_init_info.Instance = vk.instance;
    imgui_vk_init_info.PhysicalDevice = vk.physical_device;
    imgui_vk_init_info.Device = vk.device;
    imgui_vk_init_info.QueueFamily = vk.queue_family_index;
    imgui_vk_init_info.Queue = queue;
    imgui_vk_init_info.PipelineCache = 0;
    imgui_vk_init_info.DescriptorPool = imgui_descriptor_pool;
    imgui_vk_init_info.Subpass = 0;
    imgui_vk_init_info.MinImageCount = (u32)window.images.length;
    imgui_vk_init_info.ImageCount = (u32)window.images.length;
    imgui_vk_init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    // imgui_vk_init_info.ColorAttachmentFormat = window.swapchain_format;
    // imgui_vk_init_info.UseDynamicRendering = true;
    struct ImGuiCheckResult {
        static void fn(VkResult res) {
            assert(res == VK_SUCCESS);
        }
    };
    imgui_vk_init_info.CheckVkResultFn = ImGuiCheckResult::fn;

    VkAttachmentDescription attachments[1];
    attachments[0].format = window.swapchain_format;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    attachments[0].flags = 0;

    VkAttachmentReference color_reference = {};
    color_reference.attachment = 0;
    color_reference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.flags = 0;
    subpass.inputAttachmentCount = 0;
    subpass.pInputAttachments = NULL;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_reference;
    subpass.pResolveAttachments = NULL;
    subpass.pDepthStencilAttachment = NULL;
    subpass.preserveAttachmentCount = 0;
    subpass.pPreserveAttachments = NULL;

    VkSubpassDependency subpass_dependency = {};
    subpass_dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    subpass_dependency.dstSubpass = 0;
    subpass_dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    subpass_dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    subpass_dependency.srcAccessMask = 0;
    subpass_dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    subpass_dependency.dependencyFlags = 0;

    VkRenderPassCreateInfo rp_info = {};
    rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp_info.pNext = NULL;
    rp_info.attachmentCount = 1;
    rp_info.pAttachments = attachments;
    rp_info.subpassCount = 1;
    rp_info.pSubpasses = &subpass;
    rp_info.dependencyCount = 1;
    rp_info.pDependencies = &subpass_dependency;

    VkRenderPass imgui_render_pass;
    result = vkCreateRenderPass(vk.device, &rp_info, NULL, &imgui_render_pass);
    assert(result == VK_SUCCESS);

    if (!ImGui_ImplVulkan_Init(&imgui_vk_init_info, imgui_render_pass)) {
        printf("Failed to initialize Vulkan imgui backend\n");
        exit(1);
    }

    // Upload font texture.
    {
        VkCommandPool command_pool = frames[0].command_pool;
        VkCommandBuffer command_buffer= frames[0].command_buffer;

        // Reset command buffer.
        VkResult vkr = vkResetCommandPool(vk.device, command_pool, 0);
        assert(vkr == VK_SUCCESS);

        // Begin recording commands.
        VkCommandBufferBeginInfo begin_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkr = vkBeginCommandBuffer(command_buffer, &begin_info);
        assert(vkr == VK_SUCCESS);

        // Create fonts texture.
        ImGui_ImplVulkan_CreateFontsTexture(command_buffer);

        // End recording commands.
        vkr = vkEndCommandBuffer(command_buffer);
        assert(vkr == VK_SUCCESS);

        // Submit commands.
		VkPipelineStageFlags submit_stage_mask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

        VkSubmitInfo submit_info = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;

        vkr = vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
        assert(vkr == VK_SUCCESS);

        // Wait for idle.
        vkr = vkDeviceWaitIdle(vk.device);
        assert(vkr == VK_SUCCESS);
    }

    App app = {};
    app.frames = std::move(frames);
    app.framebuffers = std::move(framebuffers);
    app.window = &window;
    app.vk = &vk;
    app.queue = queue;
    app.wait_for_events = true;
    app.last_frame_timestamp = glfwGetTimerValue();

    app.data = std::move(data);
    app.plots = std::move(plots);
    app.imgui_render_pass = imgui_render_pass;

    glfwSetWindowUserPointer(window.window, &app);

    while (true) {
        if (app.wait_for_events) {
            glfwWaitEvents();
        }
        else {
            glfwPollEvents();
        }

        if (glfwWindowShouldClose(window.window)) {
            app.closed = true;
            break;
        }

        Draw(&app);
    }


    // Wait
    vkDeviceWaitIdle(vk.device);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    vmaDestroyAllocator(vk.vma);

    for (usize i = 0; i < window.image_views.length; i++) {
        VulkanFrame& frame = app.frames[i];
        vkDestroyFence(vk.device, frame.fence, 0);

        vkDestroySemaphore(vk.device, frame.acquire_semaphore, 0);
        vkDestroySemaphore(vk.device, frame.release_semaphore, 0);

        vkFreeCommandBuffers(vk.device, frame.command_pool, 1, &frame.command_buffer);
        vkDestroyCommandPool(vk.device, frame.command_pool, 0);

        if(app.framebuffers[i] != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(vk.device, app.framebuffers[i], 0);
        }
    }

    vkDestroyRenderPass(vk.device, app.imgui_render_pass, 0);

    // Window stuff
    vkDestroyDescriptorPool(vk.device, imgui_descriptor_pool, 0);
    for (usize i = 0; i < window.image_views.length; i++) {
        vkDestroyImageView(vk.device, window.image_views[i], 0);
    }
    vkDestroySwapchainKHR(vk.device, window.swapchain, 0);
    vkDestroySurfaceKHR(vk.instance, window.surface, 0);

    vkDestroyDevice(vk.device, 0);
    if(vk.debug_callback) {
        vkDestroyDebugReportCallbackEXT(vk.instance, vk.debug_callback, 0);
    }
    vkDestroyInstance(vk.instance, 0);

    //system("pause");
}
