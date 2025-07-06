#include <xpg/gui.h>
#include <xpg/log.h>

#include <implot.h>
#include <implot_internal.h>
#include <implot.cpp>
#include <implot_items.cpp>
#include <implot_demo.cpp>

#include <CLI11.hpp>

#include "roboto-medium.h"

using glm::vec3;
using glm::mat4;

using namespace xpg;

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

struct App {
    // Swapchain frames, index wraps around at the number of frames in flight.
    gfx::Context* vk;
    gfx::Window* window;
    gui::ImGuiImpl* gui;
    Array<VkFramebuffer> framebuffers; // Currently outside VulkanFrame because it's not needed when using Dyanmic rendering.
    bool wait_for_events;
    bool closed;

    //- Application data.
    ObjArray<Array<double>> data;
    std::vector<Plot> plots;
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

    gfx::SwapchainStatus swapchain_status = gfx::UpdateSwapchain(&window, vk);
    if (swapchain_status == gfx::SwapchainStatus::FAILED) {
        printf("Swapchain update failed\n");
        exit(1);
    }

    if (swapchain_status == gfx::SwapchainStatus::MINIMIZED) {
        // app->wait_for_events = true;
        return;
    }
    else if(swapchain_status == gfx::SwapchainStatus::RESIZED) {
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
        framebufferInfo.renderPass = app->gui->render_pass;
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
    gfx::Frame& frame = gfx::WaitForFrame(&window, vk);
    gfx::Result ok = gfx::AcquireImage(&frame, &window, vk);
    if (ok != gfx::Result::SUCCESS) {
        return;
    }

    // ImGui
    gui::BeginFrame();

    // ImGuiID dockspace = ImGui::DockSpaceOverViewport(NULL, ImGuiDockNodeFlags_PassthruCentralNode);
    // static bool first_frame = true;
    // ImGui::ShowDemoWindow();
    // ImPlot::ShowDemoWindow();
    {
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
                                for(s32 y = range.begin; y <= std::min(range.end, (s32)app->data.length - 1); y++) {
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
                                            Array<double>& arr_x = app->data[x];
                                            Array<double>& arr_y = app->data[y];
                                            usize count = Min(arr_x.length, arr_y.length);
                                            ImPlot::GetterXY<ImPlot::IndexerIdx<double>, ImPlot::IndexerIdx<double>> getter(
                                                ImPlot::IndexerIdx<double>(arr_x.data, (int)count),
                                                ImPlot::IndexerIdx<double>(arr_y.data, (int)count),
                                                (int)count
                                            );
                                            ImPlot::PlotLineEx(name.c_str(), getter, flags);
                                        } else {
                                            Array<double>& arr_y = app->data[y];
                                            usize count = arr_y.length;
                                            ImPlot::GetterXY<ImPlot::IndexerLin, ImPlot::IndexerIdx<double>> getter(
                                                ImPlot::IndexerLin(1, 0),
                                                ImPlot::IndexerIdx<double>(arr_y.data, (int)count),
                                                (int)count
                                            );
                                            ImPlot::PlotLineEx(name.c_str(), getter, flags);
                                        }
                                    } else {
                                        ImPlotRange range = ImPlotRange();
                                        if(plot.histogram_range.set) {
                                            range = ImPlotRange(plot.histogram_range.min, plot.histogram_range.max);
                                        }
                                        Array<double>& values = app->data[y];
                                        ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL,0.5f);
                                        ImPlot::PlotHistogram(name.c_str(), values.data, (int)values.length, plot.histogram_bins, 1.0, range, plot.histogram_flags);
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
    VkResult vkr;
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
    info.renderPass = app->gui->render_pass;
    info.framebuffer = app->framebuffers[frame.current_image_index];
    info.renderArea.extent.width = window.fb_width;
    info.renderArea.extent.height = window.fb_height;
    info.clearValueCount = ArrayCount(clear_values);
    info.pClearValues = clear_values;
    vkCmdBeginRenderPass(frame.command_buffer, &info, VK_SUBPASS_CONTENTS_INLINE);

    gui::Render(frame.command_buffer);

    vkCmdEndRenderPass(frame.command_buffer);

    vkr = vkEndCommandBuffer(frame.command_buffer);
    assert(vkr == VK_SUCCESS);

    // Submit commands
    vkr = gfx::Submit(frame, vk, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    assert(vkr == VK_SUCCESS);

    vkr = gfx::PresentFrame(&window, &frame, vk);
    assert(vkr == VK_SUCCESS);
}

static void
Callback_Key(GLFWwindow* window, int key, int scancode, int action, int mods) {
    App* app = (App*)glfwGetWindowUserPointer(window);

    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        if (key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(app->window->window, true);
        }
    }
}

static void
Callback_WindowResize(GLFWwindow* window, int width, int height) {
    App* app = (App*)glfwGetWindowUserPointer(window);
}

static void
Callback_WindowRefresh(GLFWwindow* window) {
    App* app = (App*)glfwGetWindowUserPointer(window);
    if (app) {
        Draw(app);
    }
}

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
//   [x] Move stuff to more files, move apps in own directory
//   [x] Make XPG more standalone / clean

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
//  - lines
//  x:y             <list[<scalar>:<range>> (general syntax, x y can be omitted)
//   :y                                     (equivalent to 0:y)
//   :                                      (everything is a y, default)
//  1:3-5                                   (use 1 as x, use 3,4,5 as y)
//  1:3-                                    (use 1 as x, use all after 3 as y)
//  - style:
//  m | marker      <list[char]>            (o s d ^ v l r x p a n)
//  s | size        <int>                   (marker size in pixels)
//  w | width       <int>                   (line width in pixels)
//  c | color       <list[string]>          (red, green, blue...)
//  g | grid        <bool>                  (true,false)
//  - labels:
//  n | name        <string>                (line name)
//  T | title       <string>
//  X | xlabel      <string>
//  Y | ylabel      <string>
//  - limits:
//  xlim | xlimit   <range>                 (x axis limit, automatic otherwise)
//  ylim | ylimit   <range>                 (x axis limit, automatic otherwise)
//  - scale
//  xscale          <string>                (symlog, log, lin)
//  yscale          <string>                (symlog, log, lin)
//
// Histogram (-t) supports plot options and additionally:
// b   | bin        <int|string>            (number of bins or strategy, one of
//                                           sqrt, sturges, rice, scott)
// h   | horizontal <bool>                  (horizontal histogram)
// cdf | cumulative <bool>                  (plot cdf instead of histogram)
// norm | normalize <bool>                  (normalize histogram or cdf)
// r   | range      <range>                 (range of included values)
//
// Not implemented:
//  - layout:
//  q | quadrant    <scalar, scalar>

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


const char* doc = R"(
Plot (-p):
- lines
    x:y               <list[<scalar>:<range>]> (general syntax, x y can be omitted)
     :y                                        (equivalent to 0:y)
     :                                         (everything is a y, default)
    1:3-5                                      (use 1 as x, use 3,4,5 as y)
    1:3-                                       (use 1 as x, use all after 3 as y)
- style:
    m | marker        <list[char]>             (o s d ^ v l r x p a n)
    s | size          <int>                    (marker size in pixels)
    w | width         <int>                    (line width in pixels)
    c | color         <list[string]>           (red, green, blue...)
    g | grid          <bool>                   (true,false)
- labels:
    n | name          <string>                 (line name)
    T | title         <string>
    X | xlabel        <string>
    Y | ylabel        <string>
- limits:
    xlim | xlimit     <range>                  (x axis limit, automatic otherwise)
    ylim | ylimit     <range>                  (x axis limit, automatic otherwise)
- scale
    xscale            <string>                 (symlog, log, lin)
    yscale            <string>                 (symlog, log, lin)

Histogram (-t) supports plot options and additionally:
    b    | bin        <int|string>             (number of bins or strategy, one of
                                                sqrt, sturges, rice, scott)
    h    | horizontal <bool>                   (horizontal histogram)
    cdf  | cumulative <bool>                   (plot cdf instead of histogram)
    norm | normalize  <bool>                   (normalize histogram or cdf)
    r    | range      <range>                  (range of included values)
)";

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
    std::vector<std::string> inputs;
    args.add_option("input", inputs, "Input files or - for stdin. If not given stdin is assumed");

    std::vector<std::string> binary;
    CLI::Option* opt_binary = args.add_option("-b,--binary", binary, "Binary format");

    std::vector<std::vector<std::string>> plots_options;
    args.add_option("-p,--plot", plots_options, "Plot, followed by input range and options (see below)")->take_all()->expected(0,-1);

    std::vector<std::vector<std::string>> histogram_options;
    args.add_option("-t,--hist", histogram_options, "Histogram, followed by input range and options (see below)")->take_all()->expected(0,-1);

    int window_width = 1600;
    args.add_option("-W,--width", window_width, "Window width")->check(CLI::Range(1, 100000));

    int window_height = 900;
    args.add_option("-H,--height", window_height, "Window height")->check(CLI::Range(1, 100000));

    float font_size = 16.0f;
    args.add_option("-F,--font-size", font_size, "Font size")->check(CLI::Range(1.0f, 100.0f));

    bool light_theme = false;
    args.add_flag("-L,--light", light_theme, "Set light theme");

    bool verbose = false;
    args.add_flag("-v,--verbose", verbose, "Enable log output");

    bool enable_vulkan_validation = false;
    args.add_flag("--vulkan-validation", enable_vulkan_validation, "Enable vulkan validation layer, if available");

    args.footer(doc);

    CLI11_PARSE(args, argc, argv);

    // Configure log level
    logging::set_log_level(verbose ? logging::LogLevel::Trace : logging::LogLevel::Error);

    ObjArray<Array<double>> data;

    // Validate inputs
    if(inputs.size() == 0) {
        inputs.push_back("-");
    } else {
        bool found_stdin = false;
        for(std::string& input: inputs) {
            if (input == "-") {
                if(found_stdin) {
                    Fatal("Standard input argument \"-\" can only be passed once");
                }
                found_stdin = true;
            }
        }
    }

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

        // Allocate an array for each column.
        data.resize(formats.length * inputs.size());

        for(usize input_index = 0; input_index < inputs.size(); input_index++) {
            std::string& input = inputs[input_index];

            FILE* file;
            if(input == "-") {
                file = freopen(NULL, "rb", stdin);
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

                    data[formats.length * input_index + i].add(value);
                }
            }

            if(input != "-") {
                fclose(file);
            }
        }
    } else {
        usize columns = 0;
        for(usize input_index = 0; input_index < inputs.size(); input_index++) {
            std::string& input = inputs[input_index];

            // Parse text input
            std::string line;

            std::istream* stream;
            if(input == "-") {
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
                    // Skip whitespace
                    while(it != end && std::isspace(*it)) it++;

                    char* next_it;
                    // Parse a value
                    double val = 0.0;
                    try {
                        val = strtod(it, &next_it);
                        if(next_it == it) {
                            throw std::exception();
                        }
                        it = next_it;
                    } catch (...) {
                        Fatal("%s(%d): Failed to parse value \"%s\"\n", input.c_str(), line_number,  it);
                    }

                    if(expected_count == -1) {
                        // If this is the first row add an empty column for each element
                        data.add({});
                    } else if(count >= expected_count) {
                        // If this is not the first row we expect up to this amount of items in a row
                        Fatal("%s(%d): Expected %d columns, found more\n", input.c_str(), line_number, expected_count);
                    }

                    // Add element to column and increase column index
                    data[columns + count].add(val);
                    count += 1;
                }

                // Allow empty rows
                if(count == 0) continue;

                // Check that we have the right amount of rows after the first.
                if(expected_count == -1) {
                    expected_count = count;
                } else if(expected_count != count) {
                    Fatal("%s(%d): Expected %d columns, found %d\n", input.c_str(), line_number, expected_count, count);
                }
            }
            if(expected_count <= 0) {
                Fatal("Empty input\n");
            }
            if(input != "-") {
                delete stream;
            }

            columns += expected_count;
        }
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

    gfx::Result result;
    result = gfx::Init();
    if (result != gfx::Result::SUCCESS) {
        logging::error("plot", "Failed to initialize platform\n");
    }

    gfx::Context vk = {};
    result = gfx::CreateContext(&vk, {
        .minimum_api_version = (u32)VK_API_VERSION_1_0,
        .enable_validation_layer = enable_vulkan_validation,
    });
    if (result != gfx::Result::SUCCESS) {
        logging::error("plot", "Failed to initialize vulkan\n");
        exit(100);
    }

    gfx::Window window = {};
    result = gfx::CreateWindowWithSwapchain(&window, vk, "plot", window_width, window_height);
    if (result != gfx::Result::SUCCESS) {
        logging::error("plot", "Failed to create vulkan window\n");
        exit(100);
    }

    Array<VkFramebuffer> framebuffers(window.images.length);

    // Setup window callbacks
    glfwSetWindowSizeCallback(window.window, Callback_WindowResize);
    glfwSetWindowRefreshCallback(window.window, Callback_WindowRefresh);
    glfwSetKeyCallback(window.window, Callback_Key);

    // Initialize ImGui.
    gui::ImGuiImpl imgui_impl;
    gui::CreateImGuiImpl(&imgui_impl, window, vk, {
        .dynamic_rendering = false,
        .enable_ini_and_log_files = false,
        .additional_fonts = {
            {
                .data = ArrayView<u8>(Roboto_Medium_ttf, Roboto_Medium_ttf_len),
                .size = font_size,
            }
        },
    });
    ImPlot::CreateContext();

    if(light_theme) {
        gui::SetLightTheme();
    } else {
        gui::SetDarkTheme();
    }

    App app = {};
    app.framebuffers = move(framebuffers);
    app.window = &window;
    app.vk = &vk;
    app.wait_for_events = true;
    app.data = move(data);
    app.plots = move(plots);
    app.gui = &imgui_impl;

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

    for(usize i = 0; i < app.framebuffers.length; i++) {
        vkDestroyFramebuffer(vk.device, app.framebuffers[i], 0);
    }

    // Gui
    gui::DestroyImGuiImpl(&imgui_impl, vk);

    // Window
    gfx::DestroyWindowWithSwapchain(&window, vk);

    // Context
    gfx::DestroyContext(&vk);
}
