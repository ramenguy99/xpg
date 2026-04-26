nb::enum_<ImAxis_>(mod_implot, "Axis", nb::is_arithmetic() )
    .value("X1", ImAxis_X1)
    .value("X2", ImAxis_X2)
    .value("X3", ImAxis_X3)
    .value("Y1", ImAxis_Y1)
    .value("Y2", ImAxis_Y2)
    .value("Y3", ImAxis_Y3)
;


nb::enum_<ImPlotProp_>(mod_implot, "Prop", nb::is_arithmetic() )
    .value("LINE_COLOR", ImPlotProp_LineColor)
    .value("LINE_COLORS", ImPlotProp_LineColors)
    .value("LINE_WEIGHT", ImPlotProp_LineWeight)
    .value("FILL_COLOR", ImPlotProp_FillColor)
    .value("FILL_COLORS", ImPlotProp_FillColors)
    .value("FILL_ALPHA", ImPlotProp_FillAlpha)
    .value("MARKER", ImPlotProp_Marker)
    .value("MARKER_SIZE", ImPlotProp_MarkerSize)
    .value("MARKER_SIZES", ImPlotProp_MarkerSizes)
    .value("MARKER_LINE_COLOR", ImPlotProp_MarkerLineColor)
    .value("MARKER_LINE_COLORS", ImPlotProp_MarkerLineColors)
    .value("MARKER_FILL_COLOR", ImPlotProp_MarkerFillColor)
    .value("MARKER_FILL_COLORS", ImPlotProp_MarkerFillColors)
    .value("SIZE", ImPlotProp_Size)
    .value("OFFSET", ImPlotProp_Offset)
    .value("STRIDE", ImPlotProp_Stride)
    .value("FLAGS", ImPlotProp_Flags)
;


nb::enum_<ImPlotFlags_>(mod_implot, "PlotFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImPlotFlags_None)
    .value("NO_TITLE", ImPlotFlags_NoTitle)
    .value("NO_LEGEND", ImPlotFlags_NoLegend)
    .value("NO_MOUSE_TEXT", ImPlotFlags_NoMouseText)
    .value("NO_INPUTS", ImPlotFlags_NoInputs)
    .value("NO_MENUS", ImPlotFlags_NoMenus)
    .value("NO_BOX_SELECT", ImPlotFlags_NoBoxSelect)
    .value("NO_FRAME", ImPlotFlags_NoFrame)
    .value("EQUAL", ImPlotFlags_Equal)
    .value("CROSSHAIRS", ImPlotFlags_Crosshairs)
    .value("CANVAS_ONLY", ImPlotFlags_CanvasOnly)
;


nb::enum_<ImPlotAxisFlags_>(mod_implot, "AxisFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImPlotAxisFlags_None)
    .value("NO_LABEL", ImPlotAxisFlags_NoLabel)
    .value("NO_GRID_LINES", ImPlotAxisFlags_NoGridLines)
    .value("NO_TICK_MARKS", ImPlotAxisFlags_NoTickMarks)
    .value("NO_TICK_LABELS", ImPlotAxisFlags_NoTickLabels)
    .value("NO_INITIAL_FIT", ImPlotAxisFlags_NoInitialFit)
    .value("NO_MENUS", ImPlotAxisFlags_NoMenus)
    .value("NO_SIDE_SWITCH", ImPlotAxisFlags_NoSideSwitch)
    .value("NO_HIGHLIGHT", ImPlotAxisFlags_NoHighlight)
    .value("OPPOSITE", ImPlotAxisFlags_Opposite)
    .value("FOREGROUND", ImPlotAxisFlags_Foreground)
    .value("INVERT", ImPlotAxisFlags_Invert)
    .value("AUTO_FIT", ImPlotAxisFlags_AutoFit)
    .value("RANGE_FIT", ImPlotAxisFlags_RangeFit)
    .value("PAN_STRETCH", ImPlotAxisFlags_PanStretch)
    .value("LOCK_MIN", ImPlotAxisFlags_LockMin)
    .value("LOCK_MAX", ImPlotAxisFlags_LockMax)
    .value("LOCK", ImPlotAxisFlags_Lock)
    .value("NO_DECORATIONS", ImPlotAxisFlags_NoDecorations)
    .value("AUX_DEFAULT", ImPlotAxisFlags_AuxDefault)
;


nb::enum_<ImPlotSubplotFlags_>(mod_implot, "SubplotFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImPlotSubplotFlags_None)
    .value("NO_TITLE", ImPlotSubplotFlags_NoTitle)
    .value("NO_LEGEND", ImPlotSubplotFlags_NoLegend)
    .value("NO_MENUS", ImPlotSubplotFlags_NoMenus)
    .value("NO_RESIZE", ImPlotSubplotFlags_NoResize)
    .value("NO_ALIGN", ImPlotSubplotFlags_NoAlign)
    .value("SHARE_ITEMS", ImPlotSubplotFlags_ShareItems)
    .value("LINK_ROWS", ImPlotSubplotFlags_LinkRows)
    .value("LINK_COLS", ImPlotSubplotFlags_LinkCols)
    .value("LINK_ALL_X", ImPlotSubplotFlags_LinkAllX)
    .value("LINK_ALL_Y", ImPlotSubplotFlags_LinkAllY)
    .value("COL_MAJOR", ImPlotSubplotFlags_ColMajor)
;


nb::enum_<ImPlotLegendFlags_>(mod_implot, "LegendFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImPlotLegendFlags_None)
    .value("NO_BUTTONS", ImPlotLegendFlags_NoButtons)
    .value("NO_HIGHLIGHT_ITEM", ImPlotLegendFlags_NoHighlightItem)
    .value("NO_HIGHLIGHT_AXIS", ImPlotLegendFlags_NoHighlightAxis)
    .value("NO_MENUS", ImPlotLegendFlags_NoMenus)
    .value("OUTSIDE", ImPlotLegendFlags_Outside)
    .value("HORIZONTAL", ImPlotLegendFlags_Horizontal)
    .value("SORT", ImPlotLegendFlags_Sort)
    .value("REVERSE", ImPlotLegendFlags_Reverse)
;


nb::enum_<ImPlotMouseTextFlags_>(mod_implot, "MouseTextFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImPlotMouseTextFlags_None)
    .value("NO_AUX_AXES", ImPlotMouseTextFlags_NoAuxAxes)
    .value("NO_FORMAT", ImPlotMouseTextFlags_NoFormat)
    .value("SHOW_ALWAYS", ImPlotMouseTextFlags_ShowAlways)
;


nb::enum_<ImPlotDragToolFlags_>(mod_implot, "DragToolFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImPlotDragToolFlags_None)
    .value("NO_CURSORS", ImPlotDragToolFlags_NoCursors)
    .value("NO_FIT", ImPlotDragToolFlags_NoFit)
    .value("NO_INPUTS", ImPlotDragToolFlags_NoInputs)
    .value("DELAYED", ImPlotDragToolFlags_Delayed)
;


nb::enum_<ImPlotColormapScaleFlags_>(mod_implot, "ColormapScaleFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImPlotColormapScaleFlags_None)
    .value("NO_LABEL", ImPlotColormapScaleFlags_NoLabel)
    .value("OPPOSITE", ImPlotColormapScaleFlags_Opposite)
    .value("INVERT", ImPlotColormapScaleFlags_Invert)
;


nb::enum_<ImPlotItemFlags_>(mod_implot, "ItemFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImPlotItemFlags_None)
    .value("NO_LEGEND", ImPlotItemFlags_NoLegend)
    .value("NO_FIT", ImPlotItemFlags_NoFit)
;


nb::enum_<ImPlotLineFlags_>(mod_implot, "LineFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImPlotLineFlags_None)
    .value("SEGMENTS", ImPlotLineFlags_Segments)
    .value("LOOP", ImPlotLineFlags_Loop)
    .value("SKIP_NA_N", ImPlotLineFlags_SkipNaN)
    .value("NO_CLIP", ImPlotLineFlags_NoClip)
    .value("SHADED", ImPlotLineFlags_Shaded)
;


nb::enum_<ImPlotScatterFlags_>(mod_implot, "ScatterFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImPlotScatterFlags_None)
    .value("NO_CLIP", ImPlotScatterFlags_NoClip)
;


nb::enum_<ImPlotBubblesFlags_>(mod_implot, "BubblesFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImPlotBubblesFlags_None)
;


nb::enum_<ImPlotPolygonFlags_>(mod_implot, "PolygonFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImPlotPolygonFlags_None)
    .value("CONCAVE", ImPlotPolygonFlags_Concave)
;


nb::enum_<ImPlotStairsFlags_>(mod_implot, "StairsFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImPlotStairsFlags_None)
    .value("PRE_STEP", ImPlotStairsFlags_PreStep)
    .value("SHADED", ImPlotStairsFlags_Shaded)
;


nb::enum_<ImPlotShadedFlags_>(mod_implot, "ShadedFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImPlotShadedFlags_None)
;


nb::enum_<ImPlotBarsFlags_>(mod_implot, "BarsFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImPlotBarsFlags_None)
    .value("HORIZONTAL", ImPlotBarsFlags_Horizontal)
;


nb::enum_<ImPlotBarGroupsFlags_>(mod_implot, "BarGroupsFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImPlotBarGroupsFlags_None)
    .value("HORIZONTAL", ImPlotBarGroupsFlags_Horizontal)
    .value("STACKED", ImPlotBarGroupsFlags_Stacked)
;


nb::enum_<ImPlotErrorBarsFlags_>(mod_implot, "ErrorBarsFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImPlotErrorBarsFlags_None)
    .value("HORIZONTAL", ImPlotErrorBarsFlags_Horizontal)
;


nb::enum_<ImPlotStemsFlags_>(mod_implot, "StemsFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImPlotStemsFlags_None)
    .value("HORIZONTAL", ImPlotStemsFlags_Horizontal)
;


nb::enum_<ImPlotInfLinesFlags_>(mod_implot, "InfLinesFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImPlotInfLinesFlags_None)
    .value("HORIZONTAL", ImPlotInfLinesFlags_Horizontal)
;


nb::enum_<ImPlotPieChartFlags_>(mod_implot, "PieChartFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImPlotPieChartFlags_None)
    .value("NORMALIZE", ImPlotPieChartFlags_Normalize)
    .value("IGNORE_HIDDEN", ImPlotPieChartFlags_IgnoreHidden)
    .value("EXPLODING", ImPlotPieChartFlags_Exploding)
    .value("NO_SLICE_BORDER", ImPlotPieChartFlags_NoSliceBorder)
;


nb::enum_<ImPlotHeatmapFlags_>(mod_implot, "HeatmapFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImPlotHeatmapFlags_None)
    .value("COL_MAJOR", ImPlotHeatmapFlags_ColMajor)
;


nb::enum_<ImPlotHistogramFlags_>(mod_implot, "HistogramFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImPlotHistogramFlags_None)
    .value("HORIZONTAL", ImPlotHistogramFlags_Horizontal)
    .value("CUMULATIVE", ImPlotHistogramFlags_Cumulative)
    .value("DENSITY", ImPlotHistogramFlags_Density)
    .value("NO_OUTLIERS", ImPlotHistogramFlags_NoOutliers)
    .value("COL_MAJOR", ImPlotHistogramFlags_ColMajor)
;


nb::enum_<ImPlotDigitalFlags_>(mod_implot, "DigitalFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImPlotDigitalFlags_None)
;


nb::enum_<ImPlotImageFlags_>(mod_implot, "ImageFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImPlotImageFlags_None)
;


nb::enum_<ImPlotTextFlags_>(mod_implot, "TextFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImPlotTextFlags_None)
    .value("VERTICAL", ImPlotTextFlags_Vertical)
;


nb::enum_<ImPlotDummyFlags_>(mod_implot, "DummyFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImPlotDummyFlags_None)
;


nb::enum_<ImPlotCond_>(mod_implot, "Cond", nb::is_arithmetic() )
    .value("NONE", ImPlotCond_None)
    .value("ALWAYS", ImPlotCond_Always)
    .value("ONCE", ImPlotCond_Once)
;


nb::enum_<ImPlotCol_>(mod_implot, "Color", nb::is_arithmetic() )
    .value("FRAME_BG", ImPlotCol_FrameBg)
    .value("PLOT_BG", ImPlotCol_PlotBg)
    .value("PLOT_BORDER", ImPlotCol_PlotBorder)
    .value("LEGEND_BG", ImPlotCol_LegendBg)
    .value("LEGEND_BORDER", ImPlotCol_LegendBorder)
    .value("LEGEND_TEXT", ImPlotCol_LegendText)
    .value("TITLE_TEXT", ImPlotCol_TitleText)
    .value("INLAY_TEXT", ImPlotCol_InlayText)
    .value("AXIS_TEXT", ImPlotCol_AxisText)
    .value("AXIS_GRID", ImPlotCol_AxisGrid)
    .value("AXIS_TICK", ImPlotCol_AxisTick)
    .value("AXIS_BG", ImPlotCol_AxisBg)
    .value("AXIS_BG_HOVERED", ImPlotCol_AxisBgHovered)
    .value("AXIS_BG_ACTIVE", ImPlotCol_AxisBgActive)
    .value("SELECTION", ImPlotCol_Selection)
    .value("CROSSHAIRS", ImPlotCol_Crosshairs)
;


nb::enum_<ImPlotStyleVar_>(mod_implot, "StyleVar", nb::is_arithmetic() )
    .value("PLOT_DEFAULT_SIZE", ImPlotStyleVar_PlotDefaultSize)
    .value("PLOT_MIN_SIZE", ImPlotStyleVar_PlotMinSize)
    .value("PLOT_BORDER_SIZE", ImPlotStyleVar_PlotBorderSize)
    .value("MINOR_ALPHA", ImPlotStyleVar_MinorAlpha)
    .value("MAJOR_TICK_LEN", ImPlotStyleVar_MajorTickLen)
    .value("MINOR_TICK_LEN", ImPlotStyleVar_MinorTickLen)
    .value("MAJOR_TICK_SIZE", ImPlotStyleVar_MajorTickSize)
    .value("MINOR_TICK_SIZE", ImPlotStyleVar_MinorTickSize)
    .value("MAJOR_GRID_SIZE", ImPlotStyleVar_MajorGridSize)
    .value("MINOR_GRID_SIZE", ImPlotStyleVar_MinorGridSize)
    .value("PLOT_PADDING", ImPlotStyleVar_PlotPadding)
    .value("LABEL_PADDING", ImPlotStyleVar_LabelPadding)
    .value("LEGEND_PADDING", ImPlotStyleVar_LegendPadding)
    .value("LEGEND_INNER_PADDING", ImPlotStyleVar_LegendInnerPadding)
    .value("LEGEND_SPACING", ImPlotStyleVar_LegendSpacing)
    .value("MOUSE_POS_PADDING", ImPlotStyleVar_MousePosPadding)
    .value("ANNOTATION_PADDING", ImPlotStyleVar_AnnotationPadding)
    .value("FIT_PADDING", ImPlotStyleVar_FitPadding)
    .value("DIGITAL_PADDING", ImPlotStyleVar_DigitalPadding)
    .value("DIGITAL_SPACING", ImPlotStyleVar_DigitalSpacing)
;


nb::enum_<ImPlotScale_>(mod_implot, "Scale", nb::is_arithmetic() )
    .value("LINEAR", ImPlotScale_Linear)
    .value("TIME", ImPlotScale_Time)
    .value("LOG10", ImPlotScale_Log10)
    .value("SYM_LOG", ImPlotScale_SymLog)
;


nb::enum_<ImPlotMarker_>(mod_implot, "Marker", nb::is_arithmetic() )
    .value("NONE", ImPlotMarker_None)
    .value("AUTO", ImPlotMarker_Auto)
    .value("CIRCLE", ImPlotMarker_Circle)
    .value("SQUARE", ImPlotMarker_Square)
    .value("DIAMOND", ImPlotMarker_Diamond)
    .value("UP", ImPlotMarker_Up)
    .value("DOWN", ImPlotMarker_Down)
    .value("LEFT", ImPlotMarker_Left)
    .value("RIGHT", ImPlotMarker_Right)
    .value("CROSS", ImPlotMarker_Cross)
    .value("PLUS", ImPlotMarker_Plus)
    .value("ASTERISK", ImPlotMarker_Asterisk)
;


nb::enum_<ImPlotColormap_>(mod_implot, "Colormap", nb::is_arithmetic() )
    .value("DEEP", ImPlotColormap_Deep)
    .value("DARK", ImPlotColormap_Dark)
    .value("PASTEL", ImPlotColormap_Pastel)
    .value("PAIRED", ImPlotColormap_Paired)
    .value("VIRIDIS", ImPlotColormap_Viridis)
    .value("PLASMA", ImPlotColormap_Plasma)
    .value("HOT", ImPlotColormap_Hot)
    .value("COOL", ImPlotColormap_Cool)
    .value("PINK", ImPlotColormap_Pink)
    .value("JET", ImPlotColormap_Jet)
    .value("TWILIGHT", ImPlotColormap_Twilight)
    .value("RD_BU", ImPlotColormap_RdBu)
    .value("BR_B_G", ImPlotColormap_BrBG)
    .value("PI_Y_G", ImPlotColormap_PiYG)
    .value("SPECTRAL", ImPlotColormap_Spectral)
    .value("GREYS", ImPlotColormap_Greys)
;


nb::enum_<ImPlotLocation_>(mod_implot, "Location", nb::is_arithmetic() )
    .value("CENTER", ImPlotLocation_Center)
    .value("NORTH", ImPlotLocation_North)
    .value("SOUTH", ImPlotLocation_South)
    .value("WEST", ImPlotLocation_West)
    .value("EAST", ImPlotLocation_East)
    .value("NORTH_WEST", ImPlotLocation_NorthWest)
    .value("NORTH_EAST", ImPlotLocation_NorthEast)
    .value("SOUTH_WEST", ImPlotLocation_SouthWest)
    .value("SOUTH_EAST", ImPlotLocation_SouthEast)
;


nb::enum_<ImPlotBin_>(mod_implot, "Bin", nb::is_arithmetic() )
    .value("SQRT", ImPlotBin_Sqrt)
    .value("STURGES", ImPlotBin_Sturges)
    .value("RICE", ImPlotBin_Rice)
    .value("SCOTT", ImPlotBin_Scott)
;


nb::class_<ImPlotStyle>(mod_implot, "Style")
    .def_rw("plot_default_size", &ImPlotStyle::PlotDefaultSize)
    .def_rw("plot_min_size", &ImPlotStyle::PlotMinSize)
    .def_rw("plot_border_size", &ImPlotStyle::PlotBorderSize)
    .def_rw("minor_alpha", &ImPlotStyle::MinorAlpha)
    .def_rw("major_tick_len", &ImPlotStyle::MajorTickLen)
    .def_rw("minor_tick_len", &ImPlotStyle::MinorTickLen)
    .def_rw("major_tick_size", &ImPlotStyle::MajorTickSize)
    .def_rw("minor_tick_size", &ImPlotStyle::MinorTickSize)
    .def_rw("major_grid_size", &ImPlotStyle::MajorGridSize)
    .def_rw("minor_grid_size", &ImPlotStyle::MinorGridSize)
    .def_rw("plot_padding", &ImPlotStyle::PlotPadding)
    .def_rw("label_padding", &ImPlotStyle::LabelPadding)
    .def_rw("legend_padding", &ImPlotStyle::LegendPadding)
    .def_rw("legend_inner_padding", &ImPlotStyle::LegendInnerPadding)
    .def_rw("legend_spacing", &ImPlotStyle::LegendSpacing)
    .def_rw("mouse_pos_padding", &ImPlotStyle::MousePosPadding)
    .def_rw("annotation_padding", &ImPlotStyle::AnnotationPadding)
    .def_rw("fit_padding", &ImPlotStyle::FitPadding)
    .def_rw("digital_padding", &ImPlotStyle::DigitalPadding)
    .def_rw("digital_spacing", &ImPlotStyle::DigitalSpacing)
    .def_prop_rw("color_frame_bg", [](ImPlotStyle& s) -> ImVec4 { return s.Colors[ImPlotCol_FrameBg]; }, [](ImPlotStyle& s, ImVec4 value){ s.Colors[ImPlotCol_FrameBg] = value; })
    .def_prop_rw("color_plot_bg", [](ImPlotStyle& s) -> ImVec4 { return s.Colors[ImPlotCol_PlotBg]; }, [](ImPlotStyle& s, ImVec4 value){ s.Colors[ImPlotCol_PlotBg] = value; })
    .def_prop_rw("color_plot_border", [](ImPlotStyle& s) -> ImVec4 { return s.Colors[ImPlotCol_PlotBorder]; }, [](ImPlotStyle& s, ImVec4 value){ s.Colors[ImPlotCol_PlotBorder] = value; })
    .def_prop_rw("color_legend_bg", [](ImPlotStyle& s) -> ImVec4 { return s.Colors[ImPlotCol_LegendBg]; }, [](ImPlotStyle& s, ImVec4 value){ s.Colors[ImPlotCol_LegendBg] = value; })
    .def_prop_rw("color_legend_border", [](ImPlotStyle& s) -> ImVec4 { return s.Colors[ImPlotCol_LegendBorder]; }, [](ImPlotStyle& s, ImVec4 value){ s.Colors[ImPlotCol_LegendBorder] = value; })
    .def_prop_rw("color_legend_text", [](ImPlotStyle& s) -> ImVec4 { return s.Colors[ImPlotCol_LegendText]; }, [](ImPlotStyle& s, ImVec4 value){ s.Colors[ImPlotCol_LegendText] = value; })
    .def_prop_rw("color_title_text", [](ImPlotStyle& s) -> ImVec4 { return s.Colors[ImPlotCol_TitleText]; }, [](ImPlotStyle& s, ImVec4 value){ s.Colors[ImPlotCol_TitleText] = value; })
    .def_prop_rw("color_inlay_text", [](ImPlotStyle& s) -> ImVec4 { return s.Colors[ImPlotCol_InlayText]; }, [](ImPlotStyle& s, ImVec4 value){ s.Colors[ImPlotCol_InlayText] = value; })
    .def_prop_rw("color_axis_text", [](ImPlotStyle& s) -> ImVec4 { return s.Colors[ImPlotCol_AxisText]; }, [](ImPlotStyle& s, ImVec4 value){ s.Colors[ImPlotCol_AxisText] = value; })
    .def_prop_rw("color_axis_grid", [](ImPlotStyle& s) -> ImVec4 { return s.Colors[ImPlotCol_AxisGrid]; }, [](ImPlotStyle& s, ImVec4 value){ s.Colors[ImPlotCol_AxisGrid] = value; })
    .def_prop_rw("color_axis_tick", [](ImPlotStyle& s) -> ImVec4 { return s.Colors[ImPlotCol_AxisTick]; }, [](ImPlotStyle& s, ImVec4 value){ s.Colors[ImPlotCol_AxisTick] = value; })
    .def_prop_rw("color_axis_bg", [](ImPlotStyle& s) -> ImVec4 { return s.Colors[ImPlotCol_AxisBg]; }, [](ImPlotStyle& s, ImVec4 value){ s.Colors[ImPlotCol_AxisBg] = value; })
    .def_prop_rw("color_axis_bg_hovered", [](ImPlotStyle& s) -> ImVec4 { return s.Colors[ImPlotCol_AxisBgHovered]; }, [](ImPlotStyle& s, ImVec4 value){ s.Colors[ImPlotCol_AxisBgHovered] = value; })
    .def_prop_rw("color_axis_bg_active", [](ImPlotStyle& s) -> ImVec4 { return s.Colors[ImPlotCol_AxisBgActive]; }, [](ImPlotStyle& s, ImVec4 value){ s.Colors[ImPlotCol_AxisBgActive] = value; })
    .def_prop_rw("color_selection", [](ImPlotStyle& s) -> ImVec4 { return s.Colors[ImPlotCol_Selection]; }, [](ImPlotStyle& s, ImVec4 value){ s.Colors[ImPlotCol_Selection] = value; })
    .def_prop_rw("color_crosshairs", [](ImPlotStyle& s) -> ImVec4 { return s.Colors[ImPlotCol_Crosshairs]; }, [](ImPlotStyle& s, ImVec4 value){ s.Colors[ImPlotCol_Crosshairs] = value; })
    .def_rw("colormap", &ImPlotStyle::Colormap)
    .def_rw("use_local_time", &ImPlotStyle::UseLocalTime)
    .def_rw("use_i_s_o8601", &ImPlotStyle::UseISO8601)
    .def_rw("use24_hour_clock", &ImPlotStyle::Use24HourClock)
;

nb::class_<ImPlotInputMap>(mod_implot, "InputMap")
    .def_rw("pan", &ImPlotInputMap::Pan)
    .def_rw("pan_mod", &ImPlotInputMap::PanMod)
    .def_rw("fit", &ImPlotInputMap::Fit)
    .def_rw("select", &ImPlotInputMap::Select)
    .def_rw("select_cancel", &ImPlotInputMap::SelectCancel)
    .def_rw("select_mod", &ImPlotInputMap::SelectMod)
    .def_rw("select_horz_mod", &ImPlotInputMap::SelectHorzMod)
    .def_rw("select_vert_mod", &ImPlotInputMap::SelectVertMod)
    .def_rw("menu", &ImPlotInputMap::Menu)
    .def_rw("override_mod", &ImPlotInputMap::OverrideMod)
    .def_rw("zoom_mod", &ImPlotInputMap::ZoomMod)
    .def_rw("zoom_rate", &ImPlotInputMap::ZoomRate)
;

mod_implot.def("begin_plot",
    [] (const char* title_id, ImVec2 size, ImPlotFlags_ flags) -> bool {
        return ImPlot::BeginPlot(title_id, size, flags);
    }, nb::arg("title_id"), nb::arg("size") = nb::make_tuple(-1, 0), nb::arg("flags") = 0);

mod_implot.def("end_plot",
    [] () -> void {
        return ImPlot::EndPlot();
    });

mod_implot.def("begin_subplots",
    [] (const char* title_id, int rows, int cols, ImVec2 size, ImPlotSubplotFlags_ flags) -> bool {
        return ImPlot::BeginSubplots(title_id, rows, cols, size, flags);
    }, nb::arg("title_id"), nb::arg("rows"), nb::arg("cols"), nb::arg("size"), nb::arg("flags") = 0);

mod_implot.def("end_subplots",
    [] () -> void {
        return ImPlot::EndSubplots();
    });

mod_implot.def("setup_axis",
    [] (ImAxis_ axis, const char* label, ImPlotAxisFlags_ flags) -> void {
        return ImPlot::SetupAxis(axis, label, flags);
    }, nb::arg("axis"), nb::arg("label") = nullptr, nb::arg("flags") = 0);

mod_implot.def("setup_axis_limits",
    [] (ImAxis_ axis, float v_min, float v_max, ImPlotCond_ cond) -> void {
        return ImPlot::SetupAxisLimits(axis, v_min, v_max, cond);
    }, nb::arg("axis"), nb::arg("v_min"), nb::arg("v_max"), nb::arg("cond") = ImPlotCond_Once);

mod_implot.def("setup_axis_format",
    [] (ImAxis_ axis, const char* fmt) -> void {
        return ImPlot::SetupAxisFormat(axis, fmt);
    }, nb::arg("axis"), nb::arg("fmt"));

mod_implot.def("setup_axis_scale",
    [] (ImAxis_ axis, ImPlotScale_ scale) -> void {
        return ImPlot::SetupAxisScale(axis, scale);
    }, nb::arg("axis"), nb::arg("scale"));

mod_implot.def("setup_axis_limits_constraints",
    [] (ImAxis_ axis, float v_min, float v_max) -> void {
        return ImPlot::SetupAxisLimitsConstraints(axis, v_min, v_max);
    }, nb::arg("axis"), nb::arg("v_min"), nb::arg("v_max"));

mod_implot.def("setup_axis_zoom_constraints",
    [] (ImAxis_ axis, float z_min, float z_max) -> void {
        return ImPlot::SetupAxisZoomConstraints(axis, z_min, z_max);
    }, nb::arg("axis"), nb::arg("z_min"), nb::arg("z_max"));

mod_implot.def("setup_axes",
    [] (const char* x_label, const char* y_label, ImPlotAxisFlags_ x_flags, ImPlotAxisFlags_ y_flags) -> void {
        return ImPlot::SetupAxes(x_label, y_label, x_flags, y_flags);
    }, nb::arg("x_label"), nb::arg("y_label"), nb::arg("x_flags") = 0, nb::arg("y_flags") = 0);

mod_implot.def("setup_axes_limits",
    [] (float x_min, float x_max, float y_min, float y_max, ImPlotCond_ cond) -> void {
        return ImPlot::SetupAxesLimits(x_min, x_max, y_min, y_max, cond);
    }, nb::arg("x_min"), nb::arg("x_max"), nb::arg("y_min"), nb::arg("y_max"), nb::arg("cond") = ImPlotCond_Once);

mod_implot.def("setup_legend",
    [] (ImPlotLocation_ location, ImPlotLegendFlags_ flags) -> void {
        return ImPlot::SetupLegend(location, flags);
    }, nb::arg("location"), nb::arg("flags") = 0);

mod_implot.def("setup_mouse_text",
    [] (ImPlotLocation_ location, ImPlotMouseTextFlags_ flags) -> void {
        return ImPlot::SetupMouseText(location, flags);
    }, nb::arg("location"), nb::arg("flags") = 0);

mod_implot.def("setup_finish",
    [] () -> void {
        return ImPlot::SetupFinish();
    });

mod_implot.def("set_next_axis_limits",
    [] (ImAxis_ axis, float v_min, float v_max, ImPlotCond_ cond) -> void {
        return ImPlot::SetNextAxisLimits(axis, v_min, v_max, cond);
    }, nb::arg("axis"), nb::arg("v_min"), nb::arg("v_max"), nb::arg("cond") = ImPlotCond_Once);

mod_implot.def("set_next_axis_to_fit",
    [] (ImAxis_ axis) -> void {
        return ImPlot::SetNextAxisToFit(axis);
    }, nb::arg("axis"));

mod_implot.def("set_next_axes_limits",
    [] (float x_min, float x_max, float y_min, float y_max, ImPlotCond_ cond) -> void {
        return ImPlot::SetNextAxesLimits(x_min, x_max, y_min, y_max, cond);
    }, nb::arg("x_min"), nb::arg("x_max"), nb::arg("y_min"), nb::arg("y_max"), nb::arg("cond") = ImPlotCond_Once);

mod_implot.def("set_next_axes_to_fit",
    [] () -> void {
        return ImPlot::SetNextAxesToFit();
    });

mod_implot.def("plot_line",
    [] (nb::str label_id, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> values, double xscale, double xstart, ImVec4 LineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> LineColors, float LineWeight, float FillAlpha, ImPlotMarker Marker, float MarkerSize, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, float, nb::device::cpu>> MarkerSizes, ImVec4 MarkerLineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> MarkerLineColors, ImVec4 MarkerFillColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> MarkerFillColors, ImPlotLineFlags_ flags) {
        size_t stride = values.stride(0);
        size_t count = values.shape(0);
        nb::dlpack::dtype dtype = values.dtype();
        ImPlotSpec spec;
        spec.Flags = flags;
        spec.LineColor = LineColor;
        if (LineColors.has_value()) {
            if (count != LineColors.value().shape(0)) { nb::raise("values and line_colors array must have the same count (values = %zu, LineColors = %zu)", count, LineColors.value().shape(0)); }
            spec.LineColors = LineColors.value().data();
        }
        spec.LineWeight = LineWeight;
        spec.FillAlpha = FillAlpha;
        spec.Marker = Marker;
        spec.MarkerSize = MarkerSize;
        if (MarkerSizes.has_value()) {
            if (count != MarkerSizes.value().shape(0)) { nb::raise("values and marker_sizes array must have the same count (values = %zu, MarkerSizes = %zu)", count, MarkerSizes.value().shape(0)); }
            spec.MarkerSizes = MarkerSizes.value().data();
        }
        spec.MarkerLineColor = MarkerLineColor;
        if (MarkerLineColors.has_value()) {
            if (count != MarkerLineColors.value().shape(0)) { nb::raise("values and marker_line_colors array must have the same count (values = %zu, MarkerLineColors = %zu)", count, MarkerLineColors.value().shape(0)); }
            spec.MarkerLineColors = MarkerLineColors.value().data();
        }
        spec.MarkerFillColor = MarkerFillColor;
        if (MarkerFillColors.has_value()) {
            if (count != MarkerFillColors.value().shape(0)) { nb::raise("values and marker_fill_colors array must have the same count (values = %zu, MarkerFillColors = %zu)", count, MarkerFillColors.value().shape(0)); }
            spec.MarkerFillColors = MarkerFillColors.value().data();
        }
        switch (dtype.code) {
            case (int)nb::dlpack::dtype_code::Int: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotLine(label_id.c_str(), (ImS8*)values.data(), count, xscale, xstart, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotLine(label_id.c_str(), (ImS16*)values.data(), count, xscale, xstart, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotLine(label_id.c_str(), (ImS32*)values.data(), count, xscale, xstart, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotLine(label_id.c_str(), (ImS64*)values.data(), count, xscale, xstart, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::UInt: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotLine(label_id.c_str(), (ImU8*)values.data(), count, xscale, xstart, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotLine(label_id.c_str(), (ImU16*)values.data(), count, xscale, xstart, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotLine(label_id.c_str(), (ImU32*)values.data(), count, xscale, xstart, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotLine(label_id.c_str(), (ImU64*)values.data(), count, xscale, xstart, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::Float: {
                switch (dtype.bits) {
                    case 32: spec.Stride = stride * 4; ImPlot::PlotLine(label_id.c_str(), (float*)values.data(), count, xscale, xstart, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotLine(label_id.c_str(), (double*)values.data(), count, xscale, xstart, spec); break;
                }
            } break;
            default: { nb::raise("Invalid array dtype for parameter \"values\" with code %u and %u bits", dtype.code, dtype.bits); } break;
        }
    }, nb::arg("label_id"), nb::arg("values"), nb::arg("xscale") = 1, nb::arg("xstart") = 0, nb::arg("line_color") = IMPLOT_AUTO_COL, nb::arg("line_colors") = nb::none(), nb::arg("line_weight") = 1.0, nb::arg("fill_alpha") = 1.0, nb::arg("marker") = ImPlotMarker_None, nb::arg("marker_size") = 4, nb::arg("marker_sizes") = nb::none(), nb::arg("marker_line_color") = IMPLOT_AUTO_COL, nb::arg("marker_line_colors") = nb::none(), nb::arg("marker_fill_color") = IMPLOT_AUTO_COL, nb::arg("marker_fill_colors") = nb::none(), nb::arg("flags") = ImPlotLineFlags_None);

mod_implot.def("plot_line",
    [] (nb::str label_id, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> xs, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> ys, ImVec4 LineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> LineColors, float LineWeight, float FillAlpha, ImPlotMarker Marker, float MarkerSize, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, float, nb::device::cpu>> MarkerSizes, ImVec4 MarkerLineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> MarkerLineColors, ImVec4 MarkerFillColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> MarkerFillColors, ImPlotLineFlags_ flags) {
        size_t stride = xs.stride(0);
        size_t count = xs.shape(0);
        nb::dlpack::dtype dtype = xs.dtype();
        if (dtype != ys.dtype()) { nb::raise("xs and ys array must have the same dtype (xs = %s%u, ys = %s%u)", dtype_code_to_str(dtype.bits), dtype.bits, dtype_code_to_str(ys.dtype().code), ys.dtype().bits); }
        if (count != ys.shape(0)) { nb::raise("xs and ys array must have the same count (xs = %zu, ys = %zu)", count, ys.shape(0)); }
        if (stride != ys.stride(0)) { nb::raise("xs and ys array must have the same stride (xs = %zu, ys = %zu)", stride, ys.stride(0)); }
        ImPlotSpec spec;
        spec.Flags = flags;
        spec.LineColor = LineColor;
        if (LineColors.has_value()) {
            if (count != LineColors.value().shape(0)) { nb::raise("xs and line_colors array must have the same count (xs = %zu, LineColors = %zu)", count, LineColors.value().shape(0)); }
            spec.LineColors = LineColors.value().data();
        }
        spec.LineWeight = LineWeight;
        spec.FillAlpha = FillAlpha;
        spec.Marker = Marker;
        spec.MarkerSize = MarkerSize;
        if (MarkerSizes.has_value()) {
            if (count != MarkerSizes.value().shape(0)) { nb::raise("xs and marker_sizes array must have the same count (xs = %zu, MarkerSizes = %zu)", count, MarkerSizes.value().shape(0)); }
            spec.MarkerSizes = MarkerSizes.value().data();
        }
        spec.MarkerLineColor = MarkerLineColor;
        if (MarkerLineColors.has_value()) {
            if (count != MarkerLineColors.value().shape(0)) { nb::raise("xs and marker_line_colors array must have the same count (xs = %zu, MarkerLineColors = %zu)", count, MarkerLineColors.value().shape(0)); }
            spec.MarkerLineColors = MarkerLineColors.value().data();
        }
        spec.MarkerFillColor = MarkerFillColor;
        if (MarkerFillColors.has_value()) {
            if (count != MarkerFillColors.value().shape(0)) { nb::raise("xs and marker_fill_colors array must have the same count (xs = %zu, MarkerFillColors = %zu)", count, MarkerFillColors.value().shape(0)); }
            spec.MarkerFillColors = MarkerFillColors.value().data();
        }
        switch (dtype.code) {
            case (int)nb::dlpack::dtype_code::Int: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotLine(label_id.c_str(), (ImS8*)xs.data(), (ImS8*)ys.data(), count, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotLine(label_id.c_str(), (ImS16*)xs.data(), (ImS16*)ys.data(), count, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotLine(label_id.c_str(), (ImS32*)xs.data(), (ImS32*)ys.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotLine(label_id.c_str(), (ImS64*)xs.data(), (ImS64*)ys.data(), count, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::UInt: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotLine(label_id.c_str(), (ImU8*)xs.data(), (ImU8*)ys.data(), count, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotLine(label_id.c_str(), (ImU16*)xs.data(), (ImU16*)ys.data(), count, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotLine(label_id.c_str(), (ImU32*)xs.data(), (ImU32*)ys.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotLine(label_id.c_str(), (ImU64*)xs.data(), (ImU64*)ys.data(), count, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::Float: {
                switch (dtype.bits) {
                    case 32: spec.Stride = stride * 4; ImPlot::PlotLine(label_id.c_str(), (float*)xs.data(), (float*)ys.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotLine(label_id.c_str(), (double*)xs.data(), (double*)ys.data(), count, spec); break;
                }
            } break;
            default: { nb::raise("Invalid array dtype for parameter \"xs\" with code %u and %u bits", dtype.code, dtype.bits); } break;
        }
    }, nb::arg("label_id"), nb::arg("xs"), nb::arg("ys"), nb::arg("line_color") = IMPLOT_AUTO_COL, nb::arg("line_colors") = nb::none(), nb::arg("line_weight") = 1.0, nb::arg("fill_alpha") = 1.0, nb::arg("marker") = ImPlotMarker_None, nb::arg("marker_size") = 4, nb::arg("marker_sizes") = nb::none(), nb::arg("marker_line_color") = IMPLOT_AUTO_COL, nb::arg("marker_line_colors") = nb::none(), nb::arg("marker_fill_color") = IMPLOT_AUTO_COL, nb::arg("marker_fill_colors") = nb::none(), nb::arg("flags") = ImPlotLineFlags_None);

mod_implot.def("plot_scatter",
    [] (nb::str label_id, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> values, double xscale, double xstart, ImVec4 LineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> LineColors, float LineWeight, float FillAlpha, ImPlotMarker Marker, float MarkerSize, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, float, nb::device::cpu>> MarkerSizes, ImVec4 MarkerLineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> MarkerLineColors, ImVec4 MarkerFillColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> MarkerFillColors, ImPlotScatterFlags_ flags) {
        size_t stride = values.stride(0);
        size_t count = values.shape(0);
        nb::dlpack::dtype dtype = values.dtype();
        ImPlotSpec spec;
        spec.Flags = flags;
        spec.LineColor = LineColor;
        if (LineColors.has_value()) {
            if (count != LineColors.value().shape(0)) { nb::raise("values and line_colors array must have the same count (values = %zu, LineColors = %zu)", count, LineColors.value().shape(0)); }
            spec.LineColors = LineColors.value().data();
        }
        spec.LineWeight = LineWeight;
        spec.FillAlpha = FillAlpha;
        spec.Marker = Marker;
        spec.MarkerSize = MarkerSize;
        if (MarkerSizes.has_value()) {
            if (count != MarkerSizes.value().shape(0)) { nb::raise("values and marker_sizes array must have the same count (values = %zu, MarkerSizes = %zu)", count, MarkerSizes.value().shape(0)); }
            spec.MarkerSizes = MarkerSizes.value().data();
        }
        spec.MarkerLineColor = MarkerLineColor;
        if (MarkerLineColors.has_value()) {
            if (count != MarkerLineColors.value().shape(0)) { nb::raise("values and marker_line_colors array must have the same count (values = %zu, MarkerLineColors = %zu)", count, MarkerLineColors.value().shape(0)); }
            spec.MarkerLineColors = MarkerLineColors.value().data();
        }
        spec.MarkerFillColor = MarkerFillColor;
        if (MarkerFillColors.has_value()) {
            if (count != MarkerFillColors.value().shape(0)) { nb::raise("values and marker_fill_colors array must have the same count (values = %zu, MarkerFillColors = %zu)", count, MarkerFillColors.value().shape(0)); }
            spec.MarkerFillColors = MarkerFillColors.value().data();
        }
        switch (dtype.code) {
            case (int)nb::dlpack::dtype_code::Int: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotScatter(label_id.c_str(), (ImS8*)values.data(), count, xscale, xstart, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotScatter(label_id.c_str(), (ImS16*)values.data(), count, xscale, xstart, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotScatter(label_id.c_str(), (ImS32*)values.data(), count, xscale, xstart, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotScatter(label_id.c_str(), (ImS64*)values.data(), count, xscale, xstart, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::UInt: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotScatter(label_id.c_str(), (ImU8*)values.data(), count, xscale, xstart, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotScatter(label_id.c_str(), (ImU16*)values.data(), count, xscale, xstart, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotScatter(label_id.c_str(), (ImU32*)values.data(), count, xscale, xstart, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotScatter(label_id.c_str(), (ImU64*)values.data(), count, xscale, xstart, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::Float: {
                switch (dtype.bits) {
                    case 32: spec.Stride = stride * 4; ImPlot::PlotScatter(label_id.c_str(), (float*)values.data(), count, xscale, xstart, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotScatter(label_id.c_str(), (double*)values.data(), count, xscale, xstart, spec); break;
                }
            } break;
            default: { nb::raise("Invalid array dtype for parameter \"values\" with code %u and %u bits", dtype.code, dtype.bits); } break;
        }
    }, nb::arg("label_id"), nb::arg("values"), nb::arg("xscale") = 1, nb::arg("xstart") = 0, nb::arg("line_color") = IMPLOT_AUTO_COL, nb::arg("line_colors") = nb::none(), nb::arg("line_weight") = 1.0, nb::arg("fill_alpha") = 1.0, nb::arg("marker") = ImPlotMarker_None, nb::arg("marker_size") = 4, nb::arg("marker_sizes") = nb::none(), nb::arg("marker_line_color") = IMPLOT_AUTO_COL, nb::arg("marker_line_colors") = nb::none(), nb::arg("marker_fill_color") = IMPLOT_AUTO_COL, nb::arg("marker_fill_colors") = nb::none(), nb::arg("flags") = ImPlotScatterFlags_None);

mod_implot.def("plot_scatter",
    [] (nb::str label_id, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> xs, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> ys, ImVec4 LineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> LineColors, float LineWeight, float FillAlpha, ImPlotMarker Marker, float MarkerSize, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, float, nb::device::cpu>> MarkerSizes, ImVec4 MarkerLineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> MarkerLineColors, ImVec4 MarkerFillColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> MarkerFillColors, ImPlotScatterFlags_ flags) {
        size_t stride = xs.stride(0);
        size_t count = xs.shape(0);
        nb::dlpack::dtype dtype = xs.dtype();
        if (dtype != ys.dtype()) { nb::raise("xs and ys array must have the same dtype (xs = %s%u, ys = %s%u)", dtype_code_to_str(dtype.bits), dtype.bits, dtype_code_to_str(ys.dtype().code), ys.dtype().bits); }
        if (count != ys.shape(0)) { nb::raise("xs and ys array must have the same count (xs = %zu, ys = %zu)", count, ys.shape(0)); }
        if (stride != ys.stride(0)) { nb::raise("xs and ys array must have the same stride (xs = %zu, ys = %zu)", stride, ys.stride(0)); }
        ImPlotSpec spec;
        spec.Flags = flags;
        spec.LineColor = LineColor;
        if (LineColors.has_value()) {
            if (count != LineColors.value().shape(0)) { nb::raise("xs and line_colors array must have the same count (xs = %zu, LineColors = %zu)", count, LineColors.value().shape(0)); }
            spec.LineColors = LineColors.value().data();
        }
        spec.LineWeight = LineWeight;
        spec.FillAlpha = FillAlpha;
        spec.Marker = Marker;
        spec.MarkerSize = MarkerSize;
        if (MarkerSizes.has_value()) {
            if (count != MarkerSizes.value().shape(0)) { nb::raise("xs and marker_sizes array must have the same count (xs = %zu, MarkerSizes = %zu)", count, MarkerSizes.value().shape(0)); }
            spec.MarkerSizes = MarkerSizes.value().data();
        }
        spec.MarkerLineColor = MarkerLineColor;
        if (MarkerLineColors.has_value()) {
            if (count != MarkerLineColors.value().shape(0)) { nb::raise("xs and marker_line_colors array must have the same count (xs = %zu, MarkerLineColors = %zu)", count, MarkerLineColors.value().shape(0)); }
            spec.MarkerLineColors = MarkerLineColors.value().data();
        }
        spec.MarkerFillColor = MarkerFillColor;
        if (MarkerFillColors.has_value()) {
            if (count != MarkerFillColors.value().shape(0)) { nb::raise("xs and marker_fill_colors array must have the same count (xs = %zu, MarkerFillColors = %zu)", count, MarkerFillColors.value().shape(0)); }
            spec.MarkerFillColors = MarkerFillColors.value().data();
        }
        switch (dtype.code) {
            case (int)nb::dlpack::dtype_code::Int: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotScatter(label_id.c_str(), (ImS8*)xs.data(), (ImS8*)ys.data(), count, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotScatter(label_id.c_str(), (ImS16*)xs.data(), (ImS16*)ys.data(), count, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotScatter(label_id.c_str(), (ImS32*)xs.data(), (ImS32*)ys.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotScatter(label_id.c_str(), (ImS64*)xs.data(), (ImS64*)ys.data(), count, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::UInt: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotScatter(label_id.c_str(), (ImU8*)xs.data(), (ImU8*)ys.data(), count, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotScatter(label_id.c_str(), (ImU16*)xs.data(), (ImU16*)ys.data(), count, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotScatter(label_id.c_str(), (ImU32*)xs.data(), (ImU32*)ys.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotScatter(label_id.c_str(), (ImU64*)xs.data(), (ImU64*)ys.data(), count, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::Float: {
                switch (dtype.bits) {
                    case 32: spec.Stride = stride * 4; ImPlot::PlotScatter(label_id.c_str(), (float*)xs.data(), (float*)ys.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotScatter(label_id.c_str(), (double*)xs.data(), (double*)ys.data(), count, spec); break;
                }
            } break;
            default: { nb::raise("Invalid array dtype for parameter \"xs\" with code %u and %u bits", dtype.code, dtype.bits); } break;
        }
    }, nb::arg("label_id"), nb::arg("xs"), nb::arg("ys"), nb::arg("line_color") = IMPLOT_AUTO_COL, nb::arg("line_colors") = nb::none(), nb::arg("line_weight") = 1.0, nb::arg("fill_alpha") = 1.0, nb::arg("marker") = ImPlotMarker_None, nb::arg("marker_size") = 4, nb::arg("marker_sizes") = nb::none(), nb::arg("marker_line_color") = IMPLOT_AUTO_COL, nb::arg("marker_line_colors") = nb::none(), nb::arg("marker_fill_color") = IMPLOT_AUTO_COL, nb::arg("marker_fill_colors") = nb::none(), nb::arg("flags") = ImPlotScatterFlags_None);

mod_implot.def("plot_bubbles",
    [] (nb::str label_id, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> values, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> szs, double xscale, double xstart, ImVec4 LineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> LineColors, float LineWeight, ImVec4 FillColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> FillColors, float FillAlpha, ImPlotBubblesFlags_ flags) {
        size_t stride = values.stride(0);
        size_t count = values.shape(0);
        nb::dlpack::dtype dtype = values.dtype();
        if (dtype != szs.dtype()) { nb::raise("values and szs array must have the same dtype (values = %s%u, szs = %s%u)", dtype_code_to_str(dtype.bits), dtype.bits, dtype_code_to_str(szs.dtype().code), szs.dtype().bits); }
        if (count != szs.shape(0)) { nb::raise("values and szs array must have the same count (values = %zu, szs = %zu)", count, szs.shape(0)); }
        if (stride != szs.stride(0)) { nb::raise("values and szs array must have the same stride (values = %zu, szs = %zu)", stride, szs.stride(0)); }
        ImPlotSpec spec;
        spec.Flags = flags;
        spec.LineColor = LineColor;
        if (LineColors.has_value()) {
            if (count != LineColors.value().shape(0)) { nb::raise("values and line_colors array must have the same count (values = %zu, LineColors = %zu)", count, LineColors.value().shape(0)); }
            spec.LineColors = LineColors.value().data();
        }
        spec.LineWeight = LineWeight;
        spec.FillColor = FillColor;
        if (FillColors.has_value()) {
            if (count != FillColors.value().shape(0)) { nb::raise("values and fill_colors array must have the same count (values = %zu, FillColors = %zu)", count, FillColors.value().shape(0)); }
            spec.FillColors = FillColors.value().data();
        }
        spec.FillAlpha = FillAlpha;
        switch (dtype.code) {
            case (int)nb::dlpack::dtype_code::Int: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotBubbles(label_id.c_str(), (ImS8*)values.data(), (ImS8*)szs.data(), count, xscale, xstart, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotBubbles(label_id.c_str(), (ImS16*)values.data(), (ImS16*)szs.data(), count, xscale, xstart, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotBubbles(label_id.c_str(), (ImS32*)values.data(), (ImS32*)szs.data(), count, xscale, xstart, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotBubbles(label_id.c_str(), (ImS64*)values.data(), (ImS64*)szs.data(), count, xscale, xstart, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::UInt: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotBubbles(label_id.c_str(), (ImU8*)values.data(), (ImU8*)szs.data(), count, xscale, xstart, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotBubbles(label_id.c_str(), (ImU16*)values.data(), (ImU16*)szs.data(), count, xscale, xstart, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotBubbles(label_id.c_str(), (ImU32*)values.data(), (ImU32*)szs.data(), count, xscale, xstart, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotBubbles(label_id.c_str(), (ImU64*)values.data(), (ImU64*)szs.data(), count, xscale, xstart, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::Float: {
                switch (dtype.bits) {
                    case 32: spec.Stride = stride * 4; ImPlot::PlotBubbles(label_id.c_str(), (float*)values.data(), (float*)szs.data(), count, xscale, xstart, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotBubbles(label_id.c_str(), (double*)values.data(), (double*)szs.data(), count, xscale, xstart, spec); break;
                }
            } break;
            default: { nb::raise("Invalid array dtype for parameter \"values\" with code %u and %u bits", dtype.code, dtype.bits); } break;
        }
    }, nb::arg("label_id"), nb::arg("values"), nb::arg("szs"), nb::arg("xscale") = 1, nb::arg("xstart") = 0, nb::arg("line_color") = IMPLOT_AUTO_COL, nb::arg("line_colors") = nb::none(), nb::arg("line_weight") = 1.0, nb::arg("fill_color") = IMPLOT_AUTO_COL, nb::arg("fill_colors") = nb::none(), nb::arg("fill_alpha") = 1.0, nb::arg("flags") = ImPlotBubblesFlags_None);

mod_implot.def("plot_bubbles",
    [] (nb::str label_id, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> xs, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> ys, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> szs, ImVec4 LineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> LineColors, float LineWeight, ImVec4 FillColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> FillColors, float FillAlpha, ImPlotBubblesFlags_ flags) {
        size_t stride = xs.stride(0);
        size_t count = xs.shape(0);
        nb::dlpack::dtype dtype = xs.dtype();
        if (dtype != ys.dtype()) { nb::raise("xs and ys array must have the same dtype (xs = %s%u, ys = %s%u)", dtype_code_to_str(dtype.bits), dtype.bits, dtype_code_to_str(ys.dtype().code), ys.dtype().bits); }
        if (count != ys.shape(0)) { nb::raise("xs and ys array must have the same count (xs = %zu, ys = %zu)", count, ys.shape(0)); }
        if (stride != ys.stride(0)) { nb::raise("xs and ys array must have the same stride (xs = %zu, ys = %zu)", stride, ys.stride(0)); }
        if (dtype != szs.dtype()) { nb::raise("xs and szs array must have the same dtype (xs = %s%u, szs = %s%u)", dtype_code_to_str(dtype.bits), dtype.bits, dtype_code_to_str(szs.dtype().code), szs.dtype().bits); }
        if (count != szs.shape(0)) { nb::raise("xs and szs array must have the same count (xs = %zu, szs = %zu)", count, szs.shape(0)); }
        if (stride != szs.stride(0)) { nb::raise("xs and szs array must have the same stride (xs = %zu, szs = %zu)", stride, szs.stride(0)); }
        ImPlotSpec spec;
        spec.Flags = flags;
        spec.LineColor = LineColor;
        if (LineColors.has_value()) {
            if (count != LineColors.value().shape(0)) { nb::raise("xs and line_colors array must have the same count (xs = %zu, LineColors = %zu)", count, LineColors.value().shape(0)); }
            spec.LineColors = LineColors.value().data();
        }
        spec.LineWeight = LineWeight;
        spec.FillColor = FillColor;
        if (FillColors.has_value()) {
            if (count != FillColors.value().shape(0)) { nb::raise("xs and fill_colors array must have the same count (xs = %zu, FillColors = %zu)", count, FillColors.value().shape(0)); }
            spec.FillColors = FillColors.value().data();
        }
        spec.FillAlpha = FillAlpha;
        switch (dtype.code) {
            case (int)nb::dlpack::dtype_code::Int: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotBubbles(label_id.c_str(), (ImS8*)xs.data(), (ImS8*)ys.data(), (ImS8*)szs.data(), count, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotBubbles(label_id.c_str(), (ImS16*)xs.data(), (ImS16*)ys.data(), (ImS16*)szs.data(), count, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotBubbles(label_id.c_str(), (ImS32*)xs.data(), (ImS32*)ys.data(), (ImS32*)szs.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotBubbles(label_id.c_str(), (ImS64*)xs.data(), (ImS64*)ys.data(), (ImS64*)szs.data(), count, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::UInt: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotBubbles(label_id.c_str(), (ImU8*)xs.data(), (ImU8*)ys.data(), (ImU8*)szs.data(), count, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotBubbles(label_id.c_str(), (ImU16*)xs.data(), (ImU16*)ys.data(), (ImU16*)szs.data(), count, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotBubbles(label_id.c_str(), (ImU32*)xs.data(), (ImU32*)ys.data(), (ImU32*)szs.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotBubbles(label_id.c_str(), (ImU64*)xs.data(), (ImU64*)ys.data(), (ImU64*)szs.data(), count, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::Float: {
                switch (dtype.bits) {
                    case 32: spec.Stride = stride * 4; ImPlot::PlotBubbles(label_id.c_str(), (float*)xs.data(), (float*)ys.data(), (float*)szs.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotBubbles(label_id.c_str(), (double*)xs.data(), (double*)ys.data(), (double*)szs.data(), count, spec); break;
                }
            } break;
            default: { nb::raise("Invalid array dtype for parameter \"xs\" with code %u and %u bits", dtype.code, dtype.bits); } break;
        }
    }, nb::arg("label_id"), nb::arg("xs"), nb::arg("ys"), nb::arg("szs"), nb::arg("line_color") = IMPLOT_AUTO_COL, nb::arg("line_colors") = nb::none(), nb::arg("line_weight") = 1.0, nb::arg("fill_color") = IMPLOT_AUTO_COL, nb::arg("fill_colors") = nb::none(), nb::arg("fill_alpha") = 1.0, nb::arg("flags") = ImPlotBubblesFlags_None);

mod_implot.def("plot_polygon",
    [] (nb::str label_id, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> xs, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> ys, ImVec4 LineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> LineColors, float LineWeight, ImVec4 FillColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> FillColors, float FillAlpha, ImPlotMarker Marker, float MarkerSize, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, float, nb::device::cpu>> MarkerSizes, ImVec4 MarkerLineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> MarkerLineColors, ImVec4 MarkerFillColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> MarkerFillColors, ImPlotPolygonFlags_ flags) {
        size_t stride = xs.stride(0);
        size_t count = xs.shape(0);
        nb::dlpack::dtype dtype = xs.dtype();
        if (dtype != ys.dtype()) { nb::raise("xs and ys array must have the same dtype (xs = %s%u, ys = %s%u)", dtype_code_to_str(dtype.bits), dtype.bits, dtype_code_to_str(ys.dtype().code), ys.dtype().bits); }
        if (count != ys.shape(0)) { nb::raise("xs and ys array must have the same count (xs = %zu, ys = %zu)", count, ys.shape(0)); }
        if (stride != ys.stride(0)) { nb::raise("xs and ys array must have the same stride (xs = %zu, ys = %zu)", stride, ys.stride(0)); }
        ImPlotSpec spec;
        spec.Flags = flags;
        spec.LineColor = LineColor;
        if (LineColors.has_value()) {
            if (count != LineColors.value().shape(0)) { nb::raise("xs and line_colors array must have the same count (xs = %zu, LineColors = %zu)", count, LineColors.value().shape(0)); }
            spec.LineColors = LineColors.value().data();
        }
        spec.LineWeight = LineWeight;
        spec.FillColor = FillColor;
        if (FillColors.has_value()) {
            if (count != FillColors.value().shape(0)) { nb::raise("xs and fill_colors array must have the same count (xs = %zu, FillColors = %zu)", count, FillColors.value().shape(0)); }
            spec.FillColors = FillColors.value().data();
        }
        spec.FillAlpha = FillAlpha;
        spec.Marker = Marker;
        spec.MarkerSize = MarkerSize;
        if (MarkerSizes.has_value()) {
            if (count != MarkerSizes.value().shape(0)) { nb::raise("xs and marker_sizes array must have the same count (xs = %zu, MarkerSizes = %zu)", count, MarkerSizes.value().shape(0)); }
            spec.MarkerSizes = MarkerSizes.value().data();
        }
        spec.MarkerLineColor = MarkerLineColor;
        if (MarkerLineColors.has_value()) {
            if (count != MarkerLineColors.value().shape(0)) { nb::raise("xs and marker_line_colors array must have the same count (xs = %zu, MarkerLineColors = %zu)", count, MarkerLineColors.value().shape(0)); }
            spec.MarkerLineColors = MarkerLineColors.value().data();
        }
        spec.MarkerFillColor = MarkerFillColor;
        if (MarkerFillColors.has_value()) {
            if (count != MarkerFillColors.value().shape(0)) { nb::raise("xs and marker_fill_colors array must have the same count (xs = %zu, MarkerFillColors = %zu)", count, MarkerFillColors.value().shape(0)); }
            spec.MarkerFillColors = MarkerFillColors.value().data();
        }
        switch (dtype.code) {
            case (int)nb::dlpack::dtype_code::Int: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotPolygon(label_id.c_str(), (ImS8*)xs.data(), (ImS8*)ys.data(), count, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotPolygon(label_id.c_str(), (ImS16*)xs.data(), (ImS16*)ys.data(), count, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotPolygon(label_id.c_str(), (ImS32*)xs.data(), (ImS32*)ys.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotPolygon(label_id.c_str(), (ImS64*)xs.data(), (ImS64*)ys.data(), count, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::UInt: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotPolygon(label_id.c_str(), (ImU8*)xs.data(), (ImU8*)ys.data(), count, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotPolygon(label_id.c_str(), (ImU16*)xs.data(), (ImU16*)ys.data(), count, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotPolygon(label_id.c_str(), (ImU32*)xs.data(), (ImU32*)ys.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotPolygon(label_id.c_str(), (ImU64*)xs.data(), (ImU64*)ys.data(), count, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::Float: {
                switch (dtype.bits) {
                    case 32: spec.Stride = stride * 4; ImPlot::PlotPolygon(label_id.c_str(), (float*)xs.data(), (float*)ys.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotPolygon(label_id.c_str(), (double*)xs.data(), (double*)ys.data(), count, spec); break;
                }
            } break;
            default: { nb::raise("Invalid array dtype for parameter \"xs\" with code %u and %u bits", dtype.code, dtype.bits); } break;
        }
    }, nb::arg("label_id"), nb::arg("xs"), nb::arg("ys"), nb::arg("line_color") = IMPLOT_AUTO_COL, nb::arg("line_colors") = nb::none(), nb::arg("line_weight") = 1.0, nb::arg("fill_color") = IMPLOT_AUTO_COL, nb::arg("fill_colors") = nb::none(), nb::arg("fill_alpha") = 1.0, nb::arg("marker") = ImPlotMarker_None, nb::arg("marker_size") = 4, nb::arg("marker_sizes") = nb::none(), nb::arg("marker_line_color") = IMPLOT_AUTO_COL, nb::arg("marker_line_colors") = nb::none(), nb::arg("marker_fill_color") = IMPLOT_AUTO_COL, nb::arg("marker_fill_colors") = nb::none(), nb::arg("flags") = ImPlotPolygonFlags_None);

mod_implot.def("plot_stairs",
    [] (nb::str label_id, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> values, double xscale, double xstart, ImVec4 LineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> LineColors, float LineWeight, float FillAlpha, ImPlotMarker Marker, float MarkerSize, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, float, nb::device::cpu>> MarkerSizes, ImVec4 MarkerLineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> MarkerLineColors, ImVec4 MarkerFillColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> MarkerFillColors, ImPlotStairsFlags_ flags) {
        size_t stride = values.stride(0);
        size_t count = values.shape(0);
        nb::dlpack::dtype dtype = values.dtype();
        ImPlotSpec spec;
        spec.Flags = flags;
        spec.LineColor = LineColor;
        if (LineColors.has_value()) {
            if (count != LineColors.value().shape(0)) { nb::raise("values and line_colors array must have the same count (values = %zu, LineColors = %zu)", count, LineColors.value().shape(0)); }
            spec.LineColors = LineColors.value().data();
        }
        spec.LineWeight = LineWeight;
        spec.FillAlpha = FillAlpha;
        spec.Marker = Marker;
        spec.MarkerSize = MarkerSize;
        if (MarkerSizes.has_value()) {
            if (count != MarkerSizes.value().shape(0)) { nb::raise("values and marker_sizes array must have the same count (values = %zu, MarkerSizes = %zu)", count, MarkerSizes.value().shape(0)); }
            spec.MarkerSizes = MarkerSizes.value().data();
        }
        spec.MarkerLineColor = MarkerLineColor;
        if (MarkerLineColors.has_value()) {
            if (count != MarkerLineColors.value().shape(0)) { nb::raise("values and marker_line_colors array must have the same count (values = %zu, MarkerLineColors = %zu)", count, MarkerLineColors.value().shape(0)); }
            spec.MarkerLineColors = MarkerLineColors.value().data();
        }
        spec.MarkerFillColor = MarkerFillColor;
        if (MarkerFillColors.has_value()) {
            if (count != MarkerFillColors.value().shape(0)) { nb::raise("values and marker_fill_colors array must have the same count (values = %zu, MarkerFillColors = %zu)", count, MarkerFillColors.value().shape(0)); }
            spec.MarkerFillColors = MarkerFillColors.value().data();
        }
        switch (dtype.code) {
            case (int)nb::dlpack::dtype_code::Int: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotStairs(label_id.c_str(), (ImS8*)values.data(), count, xscale, xstart, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotStairs(label_id.c_str(), (ImS16*)values.data(), count, xscale, xstart, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotStairs(label_id.c_str(), (ImS32*)values.data(), count, xscale, xstart, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotStairs(label_id.c_str(), (ImS64*)values.data(), count, xscale, xstart, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::UInt: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotStairs(label_id.c_str(), (ImU8*)values.data(), count, xscale, xstart, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotStairs(label_id.c_str(), (ImU16*)values.data(), count, xscale, xstart, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotStairs(label_id.c_str(), (ImU32*)values.data(), count, xscale, xstart, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotStairs(label_id.c_str(), (ImU64*)values.data(), count, xscale, xstart, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::Float: {
                switch (dtype.bits) {
                    case 32: spec.Stride = stride * 4; ImPlot::PlotStairs(label_id.c_str(), (float*)values.data(), count, xscale, xstart, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotStairs(label_id.c_str(), (double*)values.data(), count, xscale, xstart, spec); break;
                }
            } break;
            default: { nb::raise("Invalid array dtype for parameter \"values\" with code %u and %u bits", dtype.code, dtype.bits); } break;
        }
    }, nb::arg("label_id"), nb::arg("values"), nb::arg("xscale") = 1, nb::arg("xstart") = 0, nb::arg("line_color") = IMPLOT_AUTO_COL, nb::arg("line_colors") = nb::none(), nb::arg("line_weight") = 1.0, nb::arg("fill_alpha") = 1.0, nb::arg("marker") = ImPlotMarker_None, nb::arg("marker_size") = 4, nb::arg("marker_sizes") = nb::none(), nb::arg("marker_line_color") = IMPLOT_AUTO_COL, nb::arg("marker_line_colors") = nb::none(), nb::arg("marker_fill_color") = IMPLOT_AUTO_COL, nb::arg("marker_fill_colors") = nb::none(), nb::arg("flags") = ImPlotStairsFlags_None);

mod_implot.def("plot_stairs",
    [] (nb::str label_id, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> xs, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> ys, ImVec4 LineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> LineColors, float LineWeight, float FillAlpha, ImPlotMarker Marker, float MarkerSize, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, float, nb::device::cpu>> MarkerSizes, ImVec4 MarkerLineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> MarkerLineColors, ImVec4 MarkerFillColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> MarkerFillColors, ImPlotStairsFlags_ flags) {
        size_t stride = xs.stride(0);
        size_t count = xs.shape(0);
        nb::dlpack::dtype dtype = xs.dtype();
        if (dtype != ys.dtype()) { nb::raise("xs and ys array must have the same dtype (xs = %s%u, ys = %s%u)", dtype_code_to_str(dtype.bits), dtype.bits, dtype_code_to_str(ys.dtype().code), ys.dtype().bits); }
        if (count != ys.shape(0)) { nb::raise("xs and ys array must have the same count (xs = %zu, ys = %zu)", count, ys.shape(0)); }
        if (stride != ys.stride(0)) { nb::raise("xs and ys array must have the same stride (xs = %zu, ys = %zu)", stride, ys.stride(0)); }
        ImPlotSpec spec;
        spec.Flags = flags;
        spec.LineColor = LineColor;
        if (LineColors.has_value()) {
            if (count != LineColors.value().shape(0)) { nb::raise("xs and line_colors array must have the same count (xs = %zu, LineColors = %zu)", count, LineColors.value().shape(0)); }
            spec.LineColors = LineColors.value().data();
        }
        spec.LineWeight = LineWeight;
        spec.FillAlpha = FillAlpha;
        spec.Marker = Marker;
        spec.MarkerSize = MarkerSize;
        if (MarkerSizes.has_value()) {
            if (count != MarkerSizes.value().shape(0)) { nb::raise("xs and marker_sizes array must have the same count (xs = %zu, MarkerSizes = %zu)", count, MarkerSizes.value().shape(0)); }
            spec.MarkerSizes = MarkerSizes.value().data();
        }
        spec.MarkerLineColor = MarkerLineColor;
        if (MarkerLineColors.has_value()) {
            if (count != MarkerLineColors.value().shape(0)) { nb::raise("xs and marker_line_colors array must have the same count (xs = %zu, MarkerLineColors = %zu)", count, MarkerLineColors.value().shape(0)); }
            spec.MarkerLineColors = MarkerLineColors.value().data();
        }
        spec.MarkerFillColor = MarkerFillColor;
        if (MarkerFillColors.has_value()) {
            if (count != MarkerFillColors.value().shape(0)) { nb::raise("xs and marker_fill_colors array must have the same count (xs = %zu, MarkerFillColors = %zu)", count, MarkerFillColors.value().shape(0)); }
            spec.MarkerFillColors = MarkerFillColors.value().data();
        }
        switch (dtype.code) {
            case (int)nb::dlpack::dtype_code::Int: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotStairs(label_id.c_str(), (ImS8*)xs.data(), (ImS8*)ys.data(), count, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotStairs(label_id.c_str(), (ImS16*)xs.data(), (ImS16*)ys.data(), count, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotStairs(label_id.c_str(), (ImS32*)xs.data(), (ImS32*)ys.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotStairs(label_id.c_str(), (ImS64*)xs.data(), (ImS64*)ys.data(), count, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::UInt: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotStairs(label_id.c_str(), (ImU8*)xs.data(), (ImU8*)ys.data(), count, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotStairs(label_id.c_str(), (ImU16*)xs.data(), (ImU16*)ys.data(), count, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotStairs(label_id.c_str(), (ImU32*)xs.data(), (ImU32*)ys.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotStairs(label_id.c_str(), (ImU64*)xs.data(), (ImU64*)ys.data(), count, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::Float: {
                switch (dtype.bits) {
                    case 32: spec.Stride = stride * 4; ImPlot::PlotStairs(label_id.c_str(), (float*)xs.data(), (float*)ys.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotStairs(label_id.c_str(), (double*)xs.data(), (double*)ys.data(), count, spec); break;
                }
            } break;
            default: { nb::raise("Invalid array dtype for parameter \"xs\" with code %u and %u bits", dtype.code, dtype.bits); } break;
        }
    }, nb::arg("label_id"), nb::arg("xs"), nb::arg("ys"), nb::arg("line_color") = IMPLOT_AUTO_COL, nb::arg("line_colors") = nb::none(), nb::arg("line_weight") = 1.0, nb::arg("fill_alpha") = 1.0, nb::arg("marker") = ImPlotMarker_None, nb::arg("marker_size") = 4, nb::arg("marker_sizes") = nb::none(), nb::arg("marker_line_color") = IMPLOT_AUTO_COL, nb::arg("marker_line_colors") = nb::none(), nb::arg("marker_fill_color") = IMPLOT_AUTO_COL, nb::arg("marker_fill_colors") = nb::none(), nb::arg("flags") = ImPlotStairsFlags_None);

mod_implot.def("plot_shaded",
    [] (nb::str label_id, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> values, double yref, double xscale, double xstart, ImVec4 LineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> LineColors, float LineWeight, ImVec4 FillColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> FillColors, float FillAlpha, ImPlotMarker Marker, float MarkerSize, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, float, nb::device::cpu>> MarkerSizes, ImVec4 MarkerLineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> MarkerLineColors, ImVec4 MarkerFillColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> MarkerFillColors, ImPlotShadedFlags_ flags) {
        size_t stride = values.stride(0);
        size_t count = values.shape(0);
        nb::dlpack::dtype dtype = values.dtype();
        ImPlotSpec spec;
        spec.Flags = flags;
        spec.LineColor = LineColor;
        if (LineColors.has_value()) {
            if (count != LineColors.value().shape(0)) { nb::raise("values and line_colors array must have the same count (values = %zu, LineColors = %zu)", count, LineColors.value().shape(0)); }
            spec.LineColors = LineColors.value().data();
        }
        spec.LineWeight = LineWeight;
        spec.FillColor = FillColor;
        if (FillColors.has_value()) {
            if (count != FillColors.value().shape(0)) { nb::raise("values and fill_colors array must have the same count (values = %zu, FillColors = %zu)", count, FillColors.value().shape(0)); }
            spec.FillColors = FillColors.value().data();
        }
        spec.FillAlpha = FillAlpha;
        spec.Marker = Marker;
        spec.MarkerSize = MarkerSize;
        if (MarkerSizes.has_value()) {
            if (count != MarkerSizes.value().shape(0)) { nb::raise("values and marker_sizes array must have the same count (values = %zu, MarkerSizes = %zu)", count, MarkerSizes.value().shape(0)); }
            spec.MarkerSizes = MarkerSizes.value().data();
        }
        spec.MarkerLineColor = MarkerLineColor;
        if (MarkerLineColors.has_value()) {
            if (count != MarkerLineColors.value().shape(0)) { nb::raise("values and marker_line_colors array must have the same count (values = %zu, MarkerLineColors = %zu)", count, MarkerLineColors.value().shape(0)); }
            spec.MarkerLineColors = MarkerLineColors.value().data();
        }
        spec.MarkerFillColor = MarkerFillColor;
        if (MarkerFillColors.has_value()) {
            if (count != MarkerFillColors.value().shape(0)) { nb::raise("values and marker_fill_colors array must have the same count (values = %zu, MarkerFillColors = %zu)", count, MarkerFillColors.value().shape(0)); }
            spec.MarkerFillColors = MarkerFillColors.value().data();
        }
        switch (dtype.code) {
            case (int)nb::dlpack::dtype_code::Int: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotShaded(label_id.c_str(), (ImS8*)values.data(), count, yref, xscale, xstart, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotShaded(label_id.c_str(), (ImS16*)values.data(), count, yref, xscale, xstart, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotShaded(label_id.c_str(), (ImS32*)values.data(), count, yref, xscale, xstart, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotShaded(label_id.c_str(), (ImS64*)values.data(), count, yref, xscale, xstart, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::UInt: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotShaded(label_id.c_str(), (ImU8*)values.data(), count, yref, xscale, xstart, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotShaded(label_id.c_str(), (ImU16*)values.data(), count, yref, xscale, xstart, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotShaded(label_id.c_str(), (ImU32*)values.data(), count, yref, xscale, xstart, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotShaded(label_id.c_str(), (ImU64*)values.data(), count, yref, xscale, xstart, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::Float: {
                switch (dtype.bits) {
                    case 32: spec.Stride = stride * 4; ImPlot::PlotShaded(label_id.c_str(), (float*)values.data(), count, yref, xscale, xstart, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotShaded(label_id.c_str(), (double*)values.data(), count, yref, xscale, xstart, spec); break;
                }
            } break;
            default: { nb::raise("Invalid array dtype for parameter \"values\" with code %u and %u bits", dtype.code, dtype.bits); } break;
        }
    }, nb::arg("label_id"), nb::arg("values"), nb::arg("yref") = 0, nb::arg("xscale") = 1, nb::arg("xstart") = 0, nb::arg("line_color") = IMPLOT_AUTO_COL, nb::arg("line_colors") = nb::none(), nb::arg("line_weight") = 1.0, nb::arg("fill_color") = IMPLOT_AUTO_COL, nb::arg("fill_colors") = nb::none(), nb::arg("fill_alpha") = 1.0, nb::arg("marker") = ImPlotMarker_None, nb::arg("marker_size") = 4, nb::arg("marker_sizes") = nb::none(), nb::arg("marker_line_color") = IMPLOT_AUTO_COL, nb::arg("marker_line_colors") = nb::none(), nb::arg("marker_fill_color") = IMPLOT_AUTO_COL, nb::arg("marker_fill_colors") = nb::none(), nb::arg("flags") = ImPlotShadedFlags_None);

mod_implot.def("plot_shaded",
    [] (nb::str label_id, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> xs, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> ys, double yref, ImVec4 LineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> LineColors, float LineWeight, ImVec4 FillColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> FillColors, float FillAlpha, ImPlotMarker Marker, float MarkerSize, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, float, nb::device::cpu>> MarkerSizes, ImVec4 MarkerLineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> MarkerLineColors, ImVec4 MarkerFillColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> MarkerFillColors, ImPlotShadedFlags_ flags) {
        size_t stride = xs.stride(0);
        size_t count = xs.shape(0);
        nb::dlpack::dtype dtype = xs.dtype();
        if (dtype != ys.dtype()) { nb::raise("xs and ys array must have the same dtype (xs = %s%u, ys = %s%u)", dtype_code_to_str(dtype.bits), dtype.bits, dtype_code_to_str(ys.dtype().code), ys.dtype().bits); }
        if (count != ys.shape(0)) { nb::raise("xs and ys array must have the same count (xs = %zu, ys = %zu)", count, ys.shape(0)); }
        if (stride != ys.stride(0)) { nb::raise("xs and ys array must have the same stride (xs = %zu, ys = %zu)", stride, ys.stride(0)); }
        ImPlotSpec spec;
        spec.Flags = flags;
        spec.LineColor = LineColor;
        if (LineColors.has_value()) {
            if (count != LineColors.value().shape(0)) { nb::raise("xs and line_colors array must have the same count (xs = %zu, LineColors = %zu)", count, LineColors.value().shape(0)); }
            spec.LineColors = LineColors.value().data();
        }
        spec.LineWeight = LineWeight;
        spec.FillColor = FillColor;
        if (FillColors.has_value()) {
            if (count != FillColors.value().shape(0)) { nb::raise("xs and fill_colors array must have the same count (xs = %zu, FillColors = %zu)", count, FillColors.value().shape(0)); }
            spec.FillColors = FillColors.value().data();
        }
        spec.FillAlpha = FillAlpha;
        spec.Marker = Marker;
        spec.MarkerSize = MarkerSize;
        if (MarkerSizes.has_value()) {
            if (count != MarkerSizes.value().shape(0)) { nb::raise("xs and marker_sizes array must have the same count (xs = %zu, MarkerSizes = %zu)", count, MarkerSizes.value().shape(0)); }
            spec.MarkerSizes = MarkerSizes.value().data();
        }
        spec.MarkerLineColor = MarkerLineColor;
        if (MarkerLineColors.has_value()) {
            if (count != MarkerLineColors.value().shape(0)) { nb::raise("xs and marker_line_colors array must have the same count (xs = %zu, MarkerLineColors = %zu)", count, MarkerLineColors.value().shape(0)); }
            spec.MarkerLineColors = MarkerLineColors.value().data();
        }
        spec.MarkerFillColor = MarkerFillColor;
        if (MarkerFillColors.has_value()) {
            if (count != MarkerFillColors.value().shape(0)) { nb::raise("xs and marker_fill_colors array must have the same count (xs = %zu, MarkerFillColors = %zu)", count, MarkerFillColors.value().shape(0)); }
            spec.MarkerFillColors = MarkerFillColors.value().data();
        }
        switch (dtype.code) {
            case (int)nb::dlpack::dtype_code::Int: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotShaded(label_id.c_str(), (ImS8*)xs.data(), (ImS8*)ys.data(), count, yref, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotShaded(label_id.c_str(), (ImS16*)xs.data(), (ImS16*)ys.data(), count, yref, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotShaded(label_id.c_str(), (ImS32*)xs.data(), (ImS32*)ys.data(), count, yref, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotShaded(label_id.c_str(), (ImS64*)xs.data(), (ImS64*)ys.data(), count, yref, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::UInt: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotShaded(label_id.c_str(), (ImU8*)xs.data(), (ImU8*)ys.data(), count, yref, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotShaded(label_id.c_str(), (ImU16*)xs.data(), (ImU16*)ys.data(), count, yref, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotShaded(label_id.c_str(), (ImU32*)xs.data(), (ImU32*)ys.data(), count, yref, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotShaded(label_id.c_str(), (ImU64*)xs.data(), (ImU64*)ys.data(), count, yref, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::Float: {
                switch (dtype.bits) {
                    case 32: spec.Stride = stride * 4; ImPlot::PlotShaded(label_id.c_str(), (float*)xs.data(), (float*)ys.data(), count, yref, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotShaded(label_id.c_str(), (double*)xs.data(), (double*)ys.data(), count, yref, spec); break;
                }
            } break;
            default: { nb::raise("Invalid array dtype for parameter \"xs\" with code %u and %u bits", dtype.code, dtype.bits); } break;
        }
    }, nb::arg("label_id"), nb::arg("xs"), nb::arg("ys"), nb::arg("yref") = 0, nb::arg("line_color") = IMPLOT_AUTO_COL, nb::arg("line_colors") = nb::none(), nb::arg("line_weight") = 1.0, nb::arg("fill_color") = IMPLOT_AUTO_COL, nb::arg("fill_colors") = nb::none(), nb::arg("fill_alpha") = 1.0, nb::arg("marker") = ImPlotMarker_None, nb::arg("marker_size") = 4, nb::arg("marker_sizes") = nb::none(), nb::arg("marker_line_color") = IMPLOT_AUTO_COL, nb::arg("marker_line_colors") = nb::none(), nb::arg("marker_fill_color") = IMPLOT_AUTO_COL, nb::arg("marker_fill_colors") = nb::none(), nb::arg("flags") = ImPlotShadedFlags_None);

mod_implot.def("plot_shaded",
    [] (nb::str label_id, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> xs, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> ys1, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> ys2, ImVec4 LineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> LineColors, float LineWeight, ImVec4 FillColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> FillColors, float FillAlpha, ImPlotMarker Marker, float MarkerSize, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, float, nb::device::cpu>> MarkerSizes, ImVec4 MarkerLineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> MarkerLineColors, ImVec4 MarkerFillColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> MarkerFillColors, ImPlotShadedFlags_ flags) {
        size_t stride = xs.stride(0);
        size_t count = xs.shape(0);
        nb::dlpack::dtype dtype = xs.dtype();
        if (dtype != ys1.dtype()) { nb::raise("xs and ys1 array must have the same dtype (xs = %s%u, ys1 = %s%u)", dtype_code_to_str(dtype.bits), dtype.bits, dtype_code_to_str(ys1.dtype().code), ys1.dtype().bits); }
        if (count != ys1.shape(0)) { nb::raise("xs and ys1 array must have the same count (xs = %zu, ys1 = %zu)", count, ys1.shape(0)); }
        if (stride != ys1.stride(0)) { nb::raise("xs and ys1 array must have the same stride (xs = %zu, ys1 = %zu)", stride, ys1.stride(0)); }
        if (dtype != ys2.dtype()) { nb::raise("xs and ys2 array must have the same dtype (xs = %s%u, ys2 = %s%u)", dtype_code_to_str(dtype.bits), dtype.bits, dtype_code_to_str(ys2.dtype().code), ys2.dtype().bits); }
        if (count != ys2.shape(0)) { nb::raise("xs and ys2 array must have the same count (xs = %zu, ys2 = %zu)", count, ys2.shape(0)); }
        if (stride != ys2.stride(0)) { nb::raise("xs and ys2 array must have the same stride (xs = %zu, ys2 = %zu)", stride, ys2.stride(0)); }
        ImPlotSpec spec;
        spec.Flags = flags;
        spec.LineColor = LineColor;
        if (LineColors.has_value()) {
            if (count != LineColors.value().shape(0)) { nb::raise("xs and line_colors array must have the same count (xs = %zu, LineColors = %zu)", count, LineColors.value().shape(0)); }
            spec.LineColors = LineColors.value().data();
        }
        spec.LineWeight = LineWeight;
        spec.FillColor = FillColor;
        if (FillColors.has_value()) {
            if (count != FillColors.value().shape(0)) { nb::raise("xs and fill_colors array must have the same count (xs = %zu, FillColors = %zu)", count, FillColors.value().shape(0)); }
            spec.FillColors = FillColors.value().data();
        }
        spec.FillAlpha = FillAlpha;
        spec.Marker = Marker;
        spec.MarkerSize = MarkerSize;
        if (MarkerSizes.has_value()) {
            if (count != MarkerSizes.value().shape(0)) { nb::raise("xs and marker_sizes array must have the same count (xs = %zu, MarkerSizes = %zu)", count, MarkerSizes.value().shape(0)); }
            spec.MarkerSizes = MarkerSizes.value().data();
        }
        spec.MarkerLineColor = MarkerLineColor;
        if (MarkerLineColors.has_value()) {
            if (count != MarkerLineColors.value().shape(0)) { nb::raise("xs and marker_line_colors array must have the same count (xs = %zu, MarkerLineColors = %zu)", count, MarkerLineColors.value().shape(0)); }
            spec.MarkerLineColors = MarkerLineColors.value().data();
        }
        spec.MarkerFillColor = MarkerFillColor;
        if (MarkerFillColors.has_value()) {
            if (count != MarkerFillColors.value().shape(0)) { nb::raise("xs and marker_fill_colors array must have the same count (xs = %zu, MarkerFillColors = %zu)", count, MarkerFillColors.value().shape(0)); }
            spec.MarkerFillColors = MarkerFillColors.value().data();
        }
        switch (dtype.code) {
            case (int)nb::dlpack::dtype_code::Int: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotShaded(label_id.c_str(), (ImS8*)xs.data(), (ImS8*)ys1.data(), (ImS8*)ys2.data(), count, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotShaded(label_id.c_str(), (ImS16*)xs.data(), (ImS16*)ys1.data(), (ImS16*)ys2.data(), count, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotShaded(label_id.c_str(), (ImS32*)xs.data(), (ImS32*)ys1.data(), (ImS32*)ys2.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotShaded(label_id.c_str(), (ImS64*)xs.data(), (ImS64*)ys1.data(), (ImS64*)ys2.data(), count, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::UInt: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotShaded(label_id.c_str(), (ImU8*)xs.data(), (ImU8*)ys1.data(), (ImU8*)ys2.data(), count, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotShaded(label_id.c_str(), (ImU16*)xs.data(), (ImU16*)ys1.data(), (ImU16*)ys2.data(), count, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotShaded(label_id.c_str(), (ImU32*)xs.data(), (ImU32*)ys1.data(), (ImU32*)ys2.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotShaded(label_id.c_str(), (ImU64*)xs.data(), (ImU64*)ys1.data(), (ImU64*)ys2.data(), count, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::Float: {
                switch (dtype.bits) {
                    case 32: spec.Stride = stride * 4; ImPlot::PlotShaded(label_id.c_str(), (float*)xs.data(), (float*)ys1.data(), (float*)ys2.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotShaded(label_id.c_str(), (double*)xs.data(), (double*)ys1.data(), (double*)ys2.data(), count, spec); break;
                }
            } break;
            default: { nb::raise("Invalid array dtype for parameter \"xs\" with code %u and %u bits", dtype.code, dtype.bits); } break;
        }
    }, nb::arg("label_id"), nb::arg("xs"), nb::arg("ys1"), nb::arg("ys2"), nb::arg("line_color") = IMPLOT_AUTO_COL, nb::arg("line_colors") = nb::none(), nb::arg("line_weight") = 1.0, nb::arg("fill_color") = IMPLOT_AUTO_COL, nb::arg("fill_colors") = nb::none(), nb::arg("fill_alpha") = 1.0, nb::arg("marker") = ImPlotMarker_None, nb::arg("marker_size") = 4, nb::arg("marker_sizes") = nb::none(), nb::arg("marker_line_color") = IMPLOT_AUTO_COL, nb::arg("marker_line_colors") = nb::none(), nb::arg("marker_fill_color") = IMPLOT_AUTO_COL, nb::arg("marker_fill_colors") = nb::none(), nb::arg("flags") = ImPlotShadedFlags_None);

mod_implot.def("plot_bars",
    [] (nb::str label_id, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> values, double barsize, double shift, ImVec4 LineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> LineColors, float LineWeight, ImVec4 FillColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> FillColors, float FillAlpha, ImPlotBarsFlags_ flags) {
        size_t stride = values.stride(0);
        size_t count = values.shape(0);
        nb::dlpack::dtype dtype = values.dtype();
        ImPlotSpec spec;
        spec.Flags = flags;
        spec.LineColor = LineColor;
        if (LineColors.has_value()) {
            if (count != LineColors.value().shape(0)) { nb::raise("values and line_colors array must have the same count (values = %zu, LineColors = %zu)", count, LineColors.value().shape(0)); }
            spec.LineColors = LineColors.value().data();
        }
        spec.LineWeight = LineWeight;
        spec.FillColor = FillColor;
        if (FillColors.has_value()) {
            if (count != FillColors.value().shape(0)) { nb::raise("values and fill_colors array must have the same count (values = %zu, FillColors = %zu)", count, FillColors.value().shape(0)); }
            spec.FillColors = FillColors.value().data();
        }
        spec.FillAlpha = FillAlpha;
        switch (dtype.code) {
            case (int)nb::dlpack::dtype_code::Int: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotBars(label_id.c_str(), (ImS8*)values.data(), count, barsize, shift, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotBars(label_id.c_str(), (ImS16*)values.data(), count, barsize, shift, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotBars(label_id.c_str(), (ImS32*)values.data(), count, barsize, shift, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotBars(label_id.c_str(), (ImS64*)values.data(), count, barsize, shift, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::UInt: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotBars(label_id.c_str(), (ImU8*)values.data(), count, barsize, shift, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotBars(label_id.c_str(), (ImU16*)values.data(), count, barsize, shift, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotBars(label_id.c_str(), (ImU32*)values.data(), count, barsize, shift, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotBars(label_id.c_str(), (ImU64*)values.data(), count, barsize, shift, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::Float: {
                switch (dtype.bits) {
                    case 32: spec.Stride = stride * 4; ImPlot::PlotBars(label_id.c_str(), (float*)values.data(), count, barsize, shift, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotBars(label_id.c_str(), (double*)values.data(), count, barsize, shift, spec); break;
                }
            } break;
            default: { nb::raise("Invalid array dtype for parameter \"values\" with code %u and %u bits", dtype.code, dtype.bits); } break;
        }
    }, nb::arg("label_id"), nb::arg("values"), nb::arg("barsize") = 0.67, nb::arg("shift") = 0, nb::arg("line_color") = IMPLOT_AUTO_COL, nb::arg("line_colors") = nb::none(), nb::arg("line_weight") = 1.0, nb::arg("fill_color") = IMPLOT_AUTO_COL, nb::arg("fill_colors") = nb::none(), nb::arg("fill_alpha") = 1.0, nb::arg("flags") = ImPlotBarsFlags_None);

mod_implot.def("plot_bars",
    [] (nb::str label_id, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> xs, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> ys, double barsize, ImVec4 LineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> LineColors, float LineWeight, ImVec4 FillColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> FillColors, float FillAlpha, ImPlotBarsFlags_ flags) {
        size_t stride = xs.stride(0);
        size_t count = xs.shape(0);
        nb::dlpack::dtype dtype = xs.dtype();
        if (dtype != ys.dtype()) { nb::raise("xs and ys array must have the same dtype (xs = %s%u, ys = %s%u)", dtype_code_to_str(dtype.bits), dtype.bits, dtype_code_to_str(ys.dtype().code), ys.dtype().bits); }
        if (count != ys.shape(0)) { nb::raise("xs and ys array must have the same count (xs = %zu, ys = %zu)", count, ys.shape(0)); }
        if (stride != ys.stride(0)) { nb::raise("xs and ys array must have the same stride (xs = %zu, ys = %zu)", stride, ys.stride(0)); }
        ImPlotSpec spec;
        spec.Flags = flags;
        spec.LineColor = LineColor;
        if (LineColors.has_value()) {
            if (count != LineColors.value().shape(0)) { nb::raise("xs and line_colors array must have the same count (xs = %zu, LineColors = %zu)", count, LineColors.value().shape(0)); }
            spec.LineColors = LineColors.value().data();
        }
        spec.LineWeight = LineWeight;
        spec.FillColor = FillColor;
        if (FillColors.has_value()) {
            if (count != FillColors.value().shape(0)) { nb::raise("xs and fill_colors array must have the same count (xs = %zu, FillColors = %zu)", count, FillColors.value().shape(0)); }
            spec.FillColors = FillColors.value().data();
        }
        spec.FillAlpha = FillAlpha;
        switch (dtype.code) {
            case (int)nb::dlpack::dtype_code::Int: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotBars(label_id.c_str(), (ImS8*)xs.data(), (ImS8*)ys.data(), count, barsize, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotBars(label_id.c_str(), (ImS16*)xs.data(), (ImS16*)ys.data(), count, barsize, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotBars(label_id.c_str(), (ImS32*)xs.data(), (ImS32*)ys.data(), count, barsize, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotBars(label_id.c_str(), (ImS64*)xs.data(), (ImS64*)ys.data(), count, barsize, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::UInt: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotBars(label_id.c_str(), (ImU8*)xs.data(), (ImU8*)ys.data(), count, barsize, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotBars(label_id.c_str(), (ImU16*)xs.data(), (ImU16*)ys.data(), count, barsize, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotBars(label_id.c_str(), (ImU32*)xs.data(), (ImU32*)ys.data(), count, barsize, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotBars(label_id.c_str(), (ImU64*)xs.data(), (ImU64*)ys.data(), count, barsize, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::Float: {
                switch (dtype.bits) {
                    case 32: spec.Stride = stride * 4; ImPlot::PlotBars(label_id.c_str(), (float*)xs.data(), (float*)ys.data(), count, barsize, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotBars(label_id.c_str(), (double*)xs.data(), (double*)ys.data(), count, barsize, spec); break;
                }
            } break;
            default: { nb::raise("Invalid array dtype for parameter \"xs\" with code %u and %u bits", dtype.code, dtype.bits); } break;
        }
    }, nb::arg("label_id"), nb::arg("xs"), nb::arg("ys"), nb::arg("barsize"), nb::arg("line_color") = IMPLOT_AUTO_COL, nb::arg("line_colors") = nb::none(), nb::arg("line_weight") = 1.0, nb::arg("fill_color") = IMPLOT_AUTO_COL, nb::arg("fill_colors") = nb::none(), nb::arg("fill_alpha") = 1.0, nb::arg("flags") = ImPlotBarsFlags_None);

mod_implot.def("plot_error_bars",
    [] (nb::str label_id, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> xs, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> ys, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> err, ImVec4 LineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> LineColors, float LineWeight, ImVec4 FillColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> FillColors, float FillAlpha, float Size, ImPlotErrorBarsFlags_ flags) {
        size_t stride = xs.stride(0);
        size_t count = xs.shape(0);
        nb::dlpack::dtype dtype = xs.dtype();
        if (dtype != ys.dtype()) { nb::raise("xs and ys array must have the same dtype (xs = %s%u, ys = %s%u)", dtype_code_to_str(dtype.bits), dtype.bits, dtype_code_to_str(ys.dtype().code), ys.dtype().bits); }
        if (count != ys.shape(0)) { nb::raise("xs and ys array must have the same count (xs = %zu, ys = %zu)", count, ys.shape(0)); }
        if (stride != ys.stride(0)) { nb::raise("xs and ys array must have the same stride (xs = %zu, ys = %zu)", stride, ys.stride(0)); }
        if (dtype != err.dtype()) { nb::raise("xs and err array must have the same dtype (xs = %s%u, err = %s%u)", dtype_code_to_str(dtype.bits), dtype.bits, dtype_code_to_str(err.dtype().code), err.dtype().bits); }
        if (count != err.shape(0)) { nb::raise("xs and err array must have the same count (xs = %zu, err = %zu)", count, err.shape(0)); }
        if (stride != err.stride(0)) { nb::raise("xs and err array must have the same stride (xs = %zu, err = %zu)", stride, err.stride(0)); }
        ImPlotSpec spec;
        spec.Flags = flags;
        spec.LineColor = LineColor;
        if (LineColors.has_value()) {
            if (count != LineColors.value().shape(0)) { nb::raise("xs and line_colors array must have the same count (xs = %zu, LineColors = %zu)", count, LineColors.value().shape(0)); }
            spec.LineColors = LineColors.value().data();
        }
        spec.LineWeight = LineWeight;
        spec.FillColor = FillColor;
        if (FillColors.has_value()) {
            if (count != FillColors.value().shape(0)) { nb::raise("xs and fill_colors array must have the same count (xs = %zu, FillColors = %zu)", count, FillColors.value().shape(0)); }
            spec.FillColors = FillColors.value().data();
        }
        spec.FillAlpha = FillAlpha;
        spec.Size = Size;
        switch (dtype.code) {
            case (int)nb::dlpack::dtype_code::Int: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotErrorBars(label_id.c_str(), (ImS8*)xs.data(), (ImS8*)ys.data(), (ImS8*)err.data(), count, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotErrorBars(label_id.c_str(), (ImS16*)xs.data(), (ImS16*)ys.data(), (ImS16*)err.data(), count, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotErrorBars(label_id.c_str(), (ImS32*)xs.data(), (ImS32*)ys.data(), (ImS32*)err.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotErrorBars(label_id.c_str(), (ImS64*)xs.data(), (ImS64*)ys.data(), (ImS64*)err.data(), count, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::UInt: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotErrorBars(label_id.c_str(), (ImU8*)xs.data(), (ImU8*)ys.data(), (ImU8*)err.data(), count, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotErrorBars(label_id.c_str(), (ImU16*)xs.data(), (ImU16*)ys.data(), (ImU16*)err.data(), count, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotErrorBars(label_id.c_str(), (ImU32*)xs.data(), (ImU32*)ys.data(), (ImU32*)err.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotErrorBars(label_id.c_str(), (ImU64*)xs.data(), (ImU64*)ys.data(), (ImU64*)err.data(), count, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::Float: {
                switch (dtype.bits) {
                    case 32: spec.Stride = stride * 4; ImPlot::PlotErrorBars(label_id.c_str(), (float*)xs.data(), (float*)ys.data(), (float*)err.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotErrorBars(label_id.c_str(), (double*)xs.data(), (double*)ys.data(), (double*)err.data(), count, spec); break;
                }
            } break;
            default: { nb::raise("Invalid array dtype for parameter \"xs\" with code %u and %u bits", dtype.code, dtype.bits); } break;
        }
    }, nb::arg("label_id"), nb::arg("xs"), nb::arg("ys"), nb::arg("err"), nb::arg("line_color") = IMPLOT_AUTO_COL, nb::arg("line_colors") = nb::none(), nb::arg("line_weight") = 1.0, nb::arg("fill_color") = IMPLOT_AUTO_COL, nb::arg("fill_colors") = nb::none(), nb::arg("fill_alpha") = 1.0, nb::arg("size") = 4, nb::arg("flags") = ImPlotErrorBarsFlags_None);

mod_implot.def("plot_error_bars",
    [] (nb::str label_id, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> xs, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> ys, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> neg, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> pos, ImVec4 LineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> LineColors, float LineWeight, ImVec4 FillColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> FillColors, float FillAlpha, float Size, ImPlotErrorBarsFlags_ flags) {
        size_t stride = xs.stride(0);
        size_t count = xs.shape(0);
        nb::dlpack::dtype dtype = xs.dtype();
        if (dtype != ys.dtype()) { nb::raise("xs and ys array must have the same dtype (xs = %s%u, ys = %s%u)", dtype_code_to_str(dtype.bits), dtype.bits, dtype_code_to_str(ys.dtype().code), ys.dtype().bits); }
        if (count != ys.shape(0)) { nb::raise("xs and ys array must have the same count (xs = %zu, ys = %zu)", count, ys.shape(0)); }
        if (stride != ys.stride(0)) { nb::raise("xs and ys array must have the same stride (xs = %zu, ys = %zu)", stride, ys.stride(0)); }
        if (dtype != neg.dtype()) { nb::raise("xs and neg array must have the same dtype (xs = %s%u, neg = %s%u)", dtype_code_to_str(dtype.bits), dtype.bits, dtype_code_to_str(neg.dtype().code), neg.dtype().bits); }
        if (count != neg.shape(0)) { nb::raise("xs and neg array must have the same count (xs = %zu, neg = %zu)", count, neg.shape(0)); }
        if (stride != neg.stride(0)) { nb::raise("xs and neg array must have the same stride (xs = %zu, neg = %zu)", stride, neg.stride(0)); }
        if (dtype != pos.dtype()) { nb::raise("xs and pos array must have the same dtype (xs = %s%u, pos = %s%u)", dtype_code_to_str(dtype.bits), dtype.bits, dtype_code_to_str(pos.dtype().code), pos.dtype().bits); }
        if (count != pos.shape(0)) { nb::raise("xs and pos array must have the same count (xs = %zu, pos = %zu)", count, pos.shape(0)); }
        if (stride != pos.stride(0)) { nb::raise("xs and pos array must have the same stride (xs = %zu, pos = %zu)", stride, pos.stride(0)); }
        ImPlotSpec spec;
        spec.Flags = flags;
        spec.LineColor = LineColor;
        if (LineColors.has_value()) {
            if (count != LineColors.value().shape(0)) { nb::raise("xs and line_colors array must have the same count (xs = %zu, LineColors = %zu)", count, LineColors.value().shape(0)); }
            spec.LineColors = LineColors.value().data();
        }
        spec.LineWeight = LineWeight;
        spec.FillColor = FillColor;
        if (FillColors.has_value()) {
            if (count != FillColors.value().shape(0)) { nb::raise("xs and fill_colors array must have the same count (xs = %zu, FillColors = %zu)", count, FillColors.value().shape(0)); }
            spec.FillColors = FillColors.value().data();
        }
        spec.FillAlpha = FillAlpha;
        spec.Size = Size;
        switch (dtype.code) {
            case (int)nb::dlpack::dtype_code::Int: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotErrorBars(label_id.c_str(), (ImS8*)xs.data(), (ImS8*)ys.data(), (ImS8*)neg.data(), (ImS8*)pos.data(), count, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotErrorBars(label_id.c_str(), (ImS16*)xs.data(), (ImS16*)ys.data(), (ImS16*)neg.data(), (ImS16*)pos.data(), count, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotErrorBars(label_id.c_str(), (ImS32*)xs.data(), (ImS32*)ys.data(), (ImS32*)neg.data(), (ImS32*)pos.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotErrorBars(label_id.c_str(), (ImS64*)xs.data(), (ImS64*)ys.data(), (ImS64*)neg.data(), (ImS64*)pos.data(), count, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::UInt: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotErrorBars(label_id.c_str(), (ImU8*)xs.data(), (ImU8*)ys.data(), (ImU8*)neg.data(), (ImU8*)pos.data(), count, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotErrorBars(label_id.c_str(), (ImU16*)xs.data(), (ImU16*)ys.data(), (ImU16*)neg.data(), (ImU16*)pos.data(), count, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotErrorBars(label_id.c_str(), (ImU32*)xs.data(), (ImU32*)ys.data(), (ImU32*)neg.data(), (ImU32*)pos.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotErrorBars(label_id.c_str(), (ImU64*)xs.data(), (ImU64*)ys.data(), (ImU64*)neg.data(), (ImU64*)pos.data(), count, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::Float: {
                switch (dtype.bits) {
                    case 32: spec.Stride = stride * 4; ImPlot::PlotErrorBars(label_id.c_str(), (float*)xs.data(), (float*)ys.data(), (float*)neg.data(), (float*)pos.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotErrorBars(label_id.c_str(), (double*)xs.data(), (double*)ys.data(), (double*)neg.data(), (double*)pos.data(), count, spec); break;
                }
            } break;
            default: { nb::raise("Invalid array dtype for parameter \"xs\" with code %u and %u bits", dtype.code, dtype.bits); } break;
        }
    }, nb::arg("label_id"), nb::arg("xs"), nb::arg("ys"), nb::arg("neg"), nb::arg("pos"), nb::arg("line_color") = IMPLOT_AUTO_COL, nb::arg("line_colors") = nb::none(), nb::arg("line_weight") = 1.0, nb::arg("fill_color") = IMPLOT_AUTO_COL, nb::arg("fill_colors") = nb::none(), nb::arg("fill_alpha") = 1.0, nb::arg("size") = 4, nb::arg("flags") = ImPlotErrorBarsFlags_None);

mod_implot.def("plot_stems",
    [] (nb::str label_id, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> values, double ref, double xscale, double xstart, ImVec4 LineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> LineColors, float LineWeight, float FillAlpha, ImPlotMarker Marker, float MarkerSize, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, float, nb::device::cpu>> MarkerSizes, ImVec4 MarkerLineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> MarkerLineColors, ImVec4 MarkerFillColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> MarkerFillColors, ImPlotStemsFlags_ flags) {
        size_t stride = values.stride(0);
        size_t count = values.shape(0);
        nb::dlpack::dtype dtype = values.dtype();
        ImPlotSpec spec;
        spec.Flags = flags;
        spec.LineColor = LineColor;
        if (LineColors.has_value()) {
            if (count != LineColors.value().shape(0)) { nb::raise("values and line_colors array must have the same count (values = %zu, LineColors = %zu)", count, LineColors.value().shape(0)); }
            spec.LineColors = LineColors.value().data();
        }
        spec.LineWeight = LineWeight;
        spec.FillAlpha = FillAlpha;
        spec.Marker = Marker;
        spec.MarkerSize = MarkerSize;
        if (MarkerSizes.has_value()) {
            if (count != MarkerSizes.value().shape(0)) { nb::raise("values and marker_sizes array must have the same count (values = %zu, MarkerSizes = %zu)", count, MarkerSizes.value().shape(0)); }
            spec.MarkerSizes = MarkerSizes.value().data();
        }
        spec.MarkerLineColor = MarkerLineColor;
        if (MarkerLineColors.has_value()) {
            if (count != MarkerLineColors.value().shape(0)) { nb::raise("values and marker_line_colors array must have the same count (values = %zu, MarkerLineColors = %zu)", count, MarkerLineColors.value().shape(0)); }
            spec.MarkerLineColors = MarkerLineColors.value().data();
        }
        spec.MarkerFillColor = MarkerFillColor;
        if (MarkerFillColors.has_value()) {
            if (count != MarkerFillColors.value().shape(0)) { nb::raise("values and marker_fill_colors array must have the same count (values = %zu, MarkerFillColors = %zu)", count, MarkerFillColors.value().shape(0)); }
            spec.MarkerFillColors = MarkerFillColors.value().data();
        }
        switch (dtype.code) {
            case (int)nb::dlpack::dtype_code::Int: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotStems(label_id.c_str(), (ImS8*)values.data(), count, ref, xscale, xstart, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotStems(label_id.c_str(), (ImS16*)values.data(), count, ref, xscale, xstart, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotStems(label_id.c_str(), (ImS32*)values.data(), count, ref, xscale, xstart, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotStems(label_id.c_str(), (ImS64*)values.data(), count, ref, xscale, xstart, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::UInt: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotStems(label_id.c_str(), (ImU8*)values.data(), count, ref, xscale, xstart, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotStems(label_id.c_str(), (ImU16*)values.data(), count, ref, xscale, xstart, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotStems(label_id.c_str(), (ImU32*)values.data(), count, ref, xscale, xstart, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotStems(label_id.c_str(), (ImU64*)values.data(), count, ref, xscale, xstart, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::Float: {
                switch (dtype.bits) {
                    case 32: spec.Stride = stride * 4; ImPlot::PlotStems(label_id.c_str(), (float*)values.data(), count, ref, xscale, xstart, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotStems(label_id.c_str(), (double*)values.data(), count, ref, xscale, xstart, spec); break;
                }
            } break;
            default: { nb::raise("Invalid array dtype for parameter \"values\" with code %u and %u bits", dtype.code, dtype.bits); } break;
        }
    }, nb::arg("label_id"), nb::arg("values"), nb::arg("ref") = 0, nb::arg("xscale") = 1, nb::arg("xstart") = 0, nb::arg("line_color") = IMPLOT_AUTO_COL, nb::arg("line_colors") = nb::none(), nb::arg("line_weight") = 1.0, nb::arg("fill_alpha") = 1.0, nb::arg("marker") = ImPlotMarker_None, nb::arg("marker_size") = 4, nb::arg("marker_sizes") = nb::none(), nb::arg("marker_line_color") = IMPLOT_AUTO_COL, nb::arg("marker_line_colors") = nb::none(), nb::arg("marker_fill_color") = IMPLOT_AUTO_COL, nb::arg("marker_fill_colors") = nb::none(), nb::arg("flags") = ImPlotStemsFlags_None);

mod_implot.def("plot_stems",
    [] (nb::str label_id, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> xs, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> ys, double ref, ImVec4 LineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> LineColors, float LineWeight, float FillAlpha, ImPlotMarker Marker, float MarkerSize, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, float, nb::device::cpu>> MarkerSizes, ImVec4 MarkerLineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> MarkerLineColors, ImVec4 MarkerFillColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> MarkerFillColors, ImPlotStemsFlags_ flags) {
        size_t stride = xs.stride(0);
        size_t count = xs.shape(0);
        nb::dlpack::dtype dtype = xs.dtype();
        if (dtype != ys.dtype()) { nb::raise("xs and ys array must have the same dtype (xs = %s%u, ys = %s%u)", dtype_code_to_str(dtype.bits), dtype.bits, dtype_code_to_str(ys.dtype().code), ys.dtype().bits); }
        if (count != ys.shape(0)) { nb::raise("xs and ys array must have the same count (xs = %zu, ys = %zu)", count, ys.shape(0)); }
        if (stride != ys.stride(0)) { nb::raise("xs and ys array must have the same stride (xs = %zu, ys = %zu)", stride, ys.stride(0)); }
        ImPlotSpec spec;
        spec.Flags = flags;
        spec.LineColor = LineColor;
        if (LineColors.has_value()) {
            if (count != LineColors.value().shape(0)) { nb::raise("xs and line_colors array must have the same count (xs = %zu, LineColors = %zu)", count, LineColors.value().shape(0)); }
            spec.LineColors = LineColors.value().data();
        }
        spec.LineWeight = LineWeight;
        spec.FillAlpha = FillAlpha;
        spec.Marker = Marker;
        spec.MarkerSize = MarkerSize;
        if (MarkerSizes.has_value()) {
            if (count != MarkerSizes.value().shape(0)) { nb::raise("xs and marker_sizes array must have the same count (xs = %zu, MarkerSizes = %zu)", count, MarkerSizes.value().shape(0)); }
            spec.MarkerSizes = MarkerSizes.value().data();
        }
        spec.MarkerLineColor = MarkerLineColor;
        if (MarkerLineColors.has_value()) {
            if (count != MarkerLineColors.value().shape(0)) { nb::raise("xs and marker_line_colors array must have the same count (xs = %zu, MarkerLineColors = %zu)", count, MarkerLineColors.value().shape(0)); }
            spec.MarkerLineColors = MarkerLineColors.value().data();
        }
        spec.MarkerFillColor = MarkerFillColor;
        if (MarkerFillColors.has_value()) {
            if (count != MarkerFillColors.value().shape(0)) { nb::raise("xs and marker_fill_colors array must have the same count (xs = %zu, MarkerFillColors = %zu)", count, MarkerFillColors.value().shape(0)); }
            spec.MarkerFillColors = MarkerFillColors.value().data();
        }
        switch (dtype.code) {
            case (int)nb::dlpack::dtype_code::Int: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotStems(label_id.c_str(), (ImS8*)xs.data(), (ImS8*)ys.data(), count, ref, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotStems(label_id.c_str(), (ImS16*)xs.data(), (ImS16*)ys.data(), count, ref, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotStems(label_id.c_str(), (ImS32*)xs.data(), (ImS32*)ys.data(), count, ref, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotStems(label_id.c_str(), (ImS64*)xs.data(), (ImS64*)ys.data(), count, ref, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::UInt: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotStems(label_id.c_str(), (ImU8*)xs.data(), (ImU8*)ys.data(), count, ref, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotStems(label_id.c_str(), (ImU16*)xs.data(), (ImU16*)ys.data(), count, ref, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotStems(label_id.c_str(), (ImU32*)xs.data(), (ImU32*)ys.data(), count, ref, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotStems(label_id.c_str(), (ImU64*)xs.data(), (ImU64*)ys.data(), count, ref, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::Float: {
                switch (dtype.bits) {
                    case 32: spec.Stride = stride * 4; ImPlot::PlotStems(label_id.c_str(), (float*)xs.data(), (float*)ys.data(), count, ref, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotStems(label_id.c_str(), (double*)xs.data(), (double*)ys.data(), count, ref, spec); break;
                }
            } break;
            default: { nb::raise("Invalid array dtype for parameter \"xs\" with code %u and %u bits", dtype.code, dtype.bits); } break;
        }
    }, nb::arg("label_id"), nb::arg("xs"), nb::arg("ys"), nb::arg("ref") = 0, nb::arg("line_color") = IMPLOT_AUTO_COL, nb::arg("line_colors") = nb::none(), nb::arg("line_weight") = 1.0, nb::arg("fill_alpha") = 1.0, nb::arg("marker") = ImPlotMarker_None, nb::arg("marker_size") = 4, nb::arg("marker_sizes") = nb::none(), nb::arg("marker_line_color") = IMPLOT_AUTO_COL, nb::arg("marker_line_colors") = nb::none(), nb::arg("marker_fill_color") = IMPLOT_AUTO_COL, nb::arg("marker_fill_colors") = nb::none(), nb::arg("flags") = ImPlotStemsFlags_None);

mod_implot.def("plot_inf_lines",
    [] (nb::str label_id, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> values, ImVec4 LineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> LineColors, float LineWeight, ImPlotInfLinesFlags_ flags) {
        size_t stride = values.stride(0);
        size_t count = values.shape(0);
        nb::dlpack::dtype dtype = values.dtype();
        ImPlotSpec spec;
        spec.Flags = flags;
        spec.LineColor = LineColor;
        if (LineColors.has_value()) {
            if (count != LineColors.value().shape(0)) { nb::raise("values and line_colors array must have the same count (values = %zu, LineColors = %zu)", count, LineColors.value().shape(0)); }
            spec.LineColors = LineColors.value().data();
        }
        spec.LineWeight = LineWeight;
        switch (dtype.code) {
            case (int)nb::dlpack::dtype_code::Int: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotInfLines(label_id.c_str(), (ImS8*)values.data(), count, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotInfLines(label_id.c_str(), (ImS16*)values.data(), count, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotInfLines(label_id.c_str(), (ImS32*)values.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotInfLines(label_id.c_str(), (ImS64*)values.data(), count, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::UInt: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotInfLines(label_id.c_str(), (ImU8*)values.data(), count, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotInfLines(label_id.c_str(), (ImU16*)values.data(), count, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotInfLines(label_id.c_str(), (ImU32*)values.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotInfLines(label_id.c_str(), (ImU64*)values.data(), count, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::Float: {
                switch (dtype.bits) {
                    case 32: spec.Stride = stride * 4; ImPlot::PlotInfLines(label_id.c_str(), (float*)values.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotInfLines(label_id.c_str(), (double*)values.data(), count, spec); break;
                }
            } break;
            default: { nb::raise("Invalid array dtype for parameter \"values\" with code %u and %u bits", dtype.code, dtype.bits); } break;
        }
    }, nb::arg("label_id"), nb::arg("values"), nb::arg("line_color") = IMPLOT_AUTO_COL, nb::arg("line_colors") = nb::none(), nb::arg("line_weight") = 1.0, nb::arg("flags") = ImPlotInfLinesFlags_None);

mod_implot.def("plot_histogram",
    [] (nb::str label_id, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> values, ImPlotBin_ bins, double bar_scale, double min_value, double max_value, ImVec4 LineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> LineColors, float LineWeight, ImVec4 FillColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> FillColors, float FillAlpha, ImPlotHistogramFlags_ flags) {
        size_t stride = values.stride(0);
        size_t count = values.shape(0);
        nb::dlpack::dtype dtype = values.dtype();
        ImPlotSpec spec;
        spec.Flags = flags;
        spec.LineColor = LineColor;
        if (LineColors.has_value()) {
            if (count != LineColors.value().shape(0)) { nb::raise("values and line_colors array must have the same count (values = %zu, LineColors = %zu)", count, LineColors.value().shape(0)); }
            spec.LineColors = LineColors.value().data();
        }
        spec.LineWeight = LineWeight;
        spec.FillColor = FillColor;
        if (FillColors.has_value()) {
            if (count != FillColors.value().shape(0)) { nb::raise("values and fill_colors array must have the same count (values = %zu, FillColors = %zu)", count, FillColors.value().shape(0)); }
            spec.FillColors = FillColors.value().data();
        }
        spec.FillAlpha = FillAlpha;
        switch (dtype.code) {
            case (int)nb::dlpack::dtype_code::Int: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotHistogram(label_id.c_str(), (ImS8*)values.data(), count, bins, bar_scale, ImPlotRange(min_value, max_value), spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotHistogram(label_id.c_str(), (ImS16*)values.data(), count, bins, bar_scale, ImPlotRange(min_value, max_value), spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotHistogram(label_id.c_str(), (ImS32*)values.data(), count, bins, bar_scale, ImPlotRange(min_value, max_value), spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotHistogram(label_id.c_str(), (ImS64*)values.data(), count, bins, bar_scale, ImPlotRange(min_value, max_value), spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::UInt: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotHistogram(label_id.c_str(), (ImU8*)values.data(), count, bins, bar_scale, ImPlotRange(min_value, max_value), spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotHistogram(label_id.c_str(), (ImU16*)values.data(), count, bins, bar_scale, ImPlotRange(min_value, max_value), spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotHistogram(label_id.c_str(), (ImU32*)values.data(), count, bins, bar_scale, ImPlotRange(min_value, max_value), spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotHistogram(label_id.c_str(), (ImU64*)values.data(), count, bins, bar_scale, ImPlotRange(min_value, max_value), spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::Float: {
                switch (dtype.bits) {
                    case 32: spec.Stride = stride * 4; ImPlot::PlotHistogram(label_id.c_str(), (float*)values.data(), count, bins, bar_scale, ImPlotRange(min_value, max_value), spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotHistogram(label_id.c_str(), (double*)values.data(), count, bins, bar_scale, ImPlotRange(min_value, max_value), spec); break;
                }
            } break;
            default: { nb::raise("Invalid array dtype for parameter \"values\" with code %u and %u bits", dtype.code, dtype.bits); } break;
        }
    }, nb::arg("label_id"), nb::arg("values"), nb::arg("bins") = ImPlotBin_Sturges, nb::arg("bar_scale") = 1, nb::arg("min_value") = 0, nb::arg("max_value") = 0, nb::arg("line_color") = IMPLOT_AUTO_COL, nb::arg("line_colors") = nb::none(), nb::arg("line_weight") = 1.0, nb::arg("fill_color") = IMPLOT_AUTO_COL, nb::arg("fill_colors") = nb::none(), nb::arg("fill_alpha") = 1.0, nb::arg("flags") = ImPlotHistogramFlags_None);

mod_implot.def("plot_histogram2_d",
    [] (nb::str label_id, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> xs, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> ys, ImPlotBin_ x_bins, ImPlotBin_ y_bins, double min_x, double max_x, double min_y, double max_y, ImVec4 LineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> LineColors, float LineWeight, ImVec4 FillColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> FillColors, float FillAlpha, ImPlotHistogramFlags_ flags) {
        size_t stride = xs.stride(0);
        size_t count = xs.shape(0);
        nb::dlpack::dtype dtype = xs.dtype();
        if (dtype != ys.dtype()) { nb::raise("xs and ys array must have the same dtype (xs = %s%u, ys = %s%u)", dtype_code_to_str(dtype.bits), dtype.bits, dtype_code_to_str(ys.dtype().code), ys.dtype().bits); }
        if (count != ys.shape(0)) { nb::raise("xs and ys array must have the same count (xs = %zu, ys = %zu)", count, ys.shape(0)); }
        if (stride != ys.stride(0)) { nb::raise("xs and ys array must have the same stride (xs = %zu, ys = %zu)", stride, ys.stride(0)); }
        ImPlotSpec spec;
        spec.Flags = flags;
        spec.LineColor = LineColor;
        if (LineColors.has_value()) {
            if (count != LineColors.value().shape(0)) { nb::raise("xs and line_colors array must have the same count (xs = %zu, LineColors = %zu)", count, LineColors.value().shape(0)); }
            spec.LineColors = LineColors.value().data();
        }
        spec.LineWeight = LineWeight;
        spec.FillColor = FillColor;
        if (FillColors.has_value()) {
            if (count != FillColors.value().shape(0)) { nb::raise("xs and fill_colors array must have the same count (xs = %zu, FillColors = %zu)", count, FillColors.value().shape(0)); }
            spec.FillColors = FillColors.value().data();
        }
        spec.FillAlpha = FillAlpha;
        switch (dtype.code) {
            case (int)nb::dlpack::dtype_code::Int: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotHistogram2D(label_id.c_str(), (ImS8*)xs.data(), (ImS8*)ys.data(), count, x_bins, y_bins, ImPlotRect(min_x, max_x, min_y, max_y), spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotHistogram2D(label_id.c_str(), (ImS16*)xs.data(), (ImS16*)ys.data(), count, x_bins, y_bins, ImPlotRect(min_x, max_x, min_y, max_y), spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotHistogram2D(label_id.c_str(), (ImS32*)xs.data(), (ImS32*)ys.data(), count, x_bins, y_bins, ImPlotRect(min_x, max_x, min_y, max_y), spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotHistogram2D(label_id.c_str(), (ImS64*)xs.data(), (ImS64*)ys.data(), count, x_bins, y_bins, ImPlotRect(min_x, max_x, min_y, max_y), spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::UInt: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotHistogram2D(label_id.c_str(), (ImU8*)xs.data(), (ImU8*)ys.data(), count, x_bins, y_bins, ImPlotRect(min_x, max_x, min_y, max_y), spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotHistogram2D(label_id.c_str(), (ImU16*)xs.data(), (ImU16*)ys.data(), count, x_bins, y_bins, ImPlotRect(min_x, max_x, min_y, max_y), spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotHistogram2D(label_id.c_str(), (ImU32*)xs.data(), (ImU32*)ys.data(), count, x_bins, y_bins, ImPlotRect(min_x, max_x, min_y, max_y), spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotHistogram2D(label_id.c_str(), (ImU64*)xs.data(), (ImU64*)ys.data(), count, x_bins, y_bins, ImPlotRect(min_x, max_x, min_y, max_y), spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::Float: {
                switch (dtype.bits) {
                    case 32: spec.Stride = stride * 4; ImPlot::PlotHistogram2D(label_id.c_str(), (float*)xs.data(), (float*)ys.data(), count, x_bins, y_bins, ImPlotRect(min_x, max_x, min_y, max_y), spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotHistogram2D(label_id.c_str(), (double*)xs.data(), (double*)ys.data(), count, x_bins, y_bins, ImPlotRect(min_x, max_x, min_y, max_y), spec); break;
                }
            } break;
            default: { nb::raise("Invalid array dtype for parameter \"xs\" with code %u and %u bits", dtype.code, dtype.bits); } break;
        }
    }, nb::arg("label_id"), nb::arg("xs"), nb::arg("ys"), nb::arg("x_bins") = ImPlotBin_Sturges, nb::arg("y_bins") = ImPlotBin_Sturges, nb::arg("min_x") = 0, nb::arg("max_x") = 0, nb::arg("min_y") = 0, nb::arg("max_y") = 0, nb::arg("line_color") = IMPLOT_AUTO_COL, nb::arg("line_colors") = nb::none(), nb::arg("line_weight") = 1.0, nb::arg("fill_color") = IMPLOT_AUTO_COL, nb::arg("fill_colors") = nb::none(), nb::arg("fill_alpha") = 1.0, nb::arg("flags") = ImPlotHistogramFlags_None);

mod_implot.def("plot_digital",
    [] (nb::str label_id, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> xs, nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> ys, ImVec4 LineColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> LineColors, float LineWeight, ImVec4 FillColor, std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, ImU32, nb::device::cpu>> FillColors, float FillAlpha, float Size, ImPlotDigitalFlags_ flags) {
        size_t stride = xs.stride(0);
        size_t count = xs.shape(0);
        nb::dlpack::dtype dtype = xs.dtype();
        if (dtype != ys.dtype()) { nb::raise("xs and ys array must have the same dtype (xs = %s%u, ys = %s%u)", dtype_code_to_str(dtype.bits), dtype.bits, dtype_code_to_str(ys.dtype().code), ys.dtype().bits); }
        if (count != ys.shape(0)) { nb::raise("xs and ys array must have the same count (xs = %zu, ys = %zu)", count, ys.shape(0)); }
        if (stride != ys.stride(0)) { nb::raise("xs and ys array must have the same stride (xs = %zu, ys = %zu)", stride, ys.stride(0)); }
        ImPlotSpec spec;
        spec.Flags = flags;
        spec.LineColor = LineColor;
        if (LineColors.has_value()) {
            if (count != LineColors.value().shape(0)) { nb::raise("xs and line_colors array must have the same count (xs = %zu, LineColors = %zu)", count, LineColors.value().shape(0)); }
            spec.LineColors = LineColors.value().data();
        }
        spec.LineWeight = LineWeight;
        spec.FillColor = FillColor;
        if (FillColors.has_value()) {
            if (count != FillColors.value().shape(0)) { nb::raise("xs and fill_colors array must have the same count (xs = %zu, FillColors = %zu)", count, FillColors.value().shape(0)); }
            spec.FillColors = FillColors.value().data();
        }
        spec.FillAlpha = FillAlpha;
        spec.Size = Size;
        switch (dtype.code) {
            case (int)nb::dlpack::dtype_code::Int: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotDigital(label_id.c_str(), (ImS8*)xs.data(), (ImS8*)ys.data(), count, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotDigital(label_id.c_str(), (ImS16*)xs.data(), (ImS16*)ys.data(), count, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotDigital(label_id.c_str(), (ImS32*)xs.data(), (ImS32*)ys.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotDigital(label_id.c_str(), (ImS64*)xs.data(), (ImS64*)ys.data(), count, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::UInt: {
                switch (dtype.bits) {
                    case  8: spec.Stride = stride * 1; ImPlot::PlotDigital(label_id.c_str(), (ImU8*)xs.data(), (ImU8*)ys.data(), count, spec); break;
                    case 16: spec.Stride = stride * 2; ImPlot::PlotDigital(label_id.c_str(), (ImU16*)xs.data(), (ImU16*)ys.data(), count, spec); break;
                    case 32: spec.Stride = stride * 4; ImPlot::PlotDigital(label_id.c_str(), (ImU32*)xs.data(), (ImU32*)ys.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotDigital(label_id.c_str(), (ImU64*)xs.data(), (ImU64*)ys.data(), count, spec); break;
                }
            } break;
            case (int)nb::dlpack::dtype_code::Float: {
                switch (dtype.bits) {
                    case 32: spec.Stride = stride * 4; ImPlot::PlotDigital(label_id.c_str(), (float*)xs.data(), (float*)ys.data(), count, spec); break;
                    case 64: spec.Stride = stride * 8; ImPlot::PlotDigital(label_id.c_str(), (double*)xs.data(), (double*)ys.data(), count, spec); break;
                }
            } break;
            default: { nb::raise("Invalid array dtype for parameter \"xs\" with code %u and %u bits", dtype.code, dtype.bits); } break;
        }
    }, nb::arg("label_id"), nb::arg("xs"), nb::arg("ys"), nb::arg("line_color") = IMPLOT_AUTO_COL, nb::arg("line_colors") = nb::none(), nb::arg("line_weight") = 1.0, nb::arg("fill_color") = IMPLOT_AUTO_COL, nb::arg("fill_colors") = nb::none(), nb::arg("fill_alpha") = 1.0, nb::arg("size") = 4, nb::arg("flags") = ImPlotDigitalFlags_None);

mod_implot.def("annotation",
    [] (float x, float y, ImVec4 col, ImVec2 pix_offset, bool clamp, bool round) -> void {
        return ImPlot::Annotation(x, y, col, pix_offset, clamp, round);
    }, nb::arg("x"), nb::arg("y"), nb::arg("col"), nb::arg("pix_offset"), nb::arg("clamp"), nb::arg("round") = false);

mod_implot.def("annotation_str",
    [] (float x, float y, ImVec4 col, ImVec2 pix_offset, bool clamp, const char* fmt) -> void {
        return ImPlot::Annotation(x, y, col, pix_offset, clamp, "%s", fmt);
    }, nb::arg("x"), nb::arg("y"), nb::arg("col"), nb::arg("pix_offset"), nb::arg("clamp"), nb::arg("fmt"));

mod_implot.def("tag_x",
    [] (float x, ImVec4 col, bool round) -> void {
        return ImPlot::TagX(x, col, round);
    }, nb::arg("x"), nb::arg("col"), nb::arg("round") = false);

mod_implot.def("tag_x_str",
    [] (float x, ImVec4 col, const char* fmt) -> void {
        return ImPlot::TagX(x, col, "%s", fmt);
    }, nb::arg("x"), nb::arg("col"), nb::arg("fmt"));

mod_implot.def("tag_y",
    [] (float y, ImVec4 col, bool round) -> void {
        return ImPlot::TagY(y, col, round);
    }, nb::arg("y"), nb::arg("col"), nb::arg("round") = false);

mod_implot.def("tag_y_str",
    [] (float y, ImVec4 col, const char* fmt) -> void {
        return ImPlot::TagY(y, col, "%s", fmt);
    }, nb::arg("y"), nb::arg("col"), nb::arg("fmt"));

mod_implot.def("set_axis",
    [] (ImAxis_ axis) -> void {
        return ImPlot::SetAxis(axis);
    }, nb::arg("axis"));

mod_implot.def("set_axes",
    [] (ImAxis_ x_axis, ImAxis_ y_axis) -> void {
        return ImPlot::SetAxes(x_axis, y_axis);
    }, nb::arg("x_axis"), nb::arg("y_axis"));

mod_implot.def("pixels_to_plot",
    [] (ImVec2 pix, ImAxis_ x_axis, ImAxis_ y_axis) -> ImPlotPoint {
        return ImPlot::PixelsToPlot(pix, x_axis, y_axis);
    }, nb::arg("pix"), nb::arg("x_axis") = IMPLOT_AUTO, nb::arg("y_axis") = IMPLOT_AUTO);

mod_implot.def("pixels_to_plot_float",
    [] (float x, float y, ImAxis_ x_axis, ImAxis_ y_axis) -> ImPlotPoint {
        return ImPlot::PixelsToPlot(x, y, x_axis, y_axis);
    }, nb::arg("x"), nb::arg("y"), nb::arg("x_axis") = IMPLOT_AUTO, nb::arg("y_axis") = IMPLOT_AUTO);

mod_implot.def("plot_to_pixels",
    [] (ImPlotPoint plt, ImAxis_ x_axis, ImAxis_ y_axis) -> ImVec2 {
        return ImPlot::PlotToPixels(plt, x_axis, y_axis);
    }, nb::arg("plt"), nb::arg("x_axis") = IMPLOT_AUTO, nb::arg("y_axis") = IMPLOT_AUTO);

mod_implot.def("plot_to_pixels_double",
    [] (float x, float y, ImAxis_ x_axis, ImAxis_ y_axis) -> ImVec2 {
        return ImPlot::PlotToPixels(x, y, x_axis, y_axis);
    }, nb::arg("x"), nb::arg("y"), nb::arg("x_axis") = IMPLOT_AUTO, nb::arg("y_axis") = IMPLOT_AUTO);

mod_implot.def("get_plot_pos",
    [] () -> ImVec2 {
        return ImPlot::GetPlotPos();
    });

mod_implot.def("get_plot_size",
    [] () -> ImVec2 {
        return ImPlot::GetPlotSize();
    });

mod_implot.def("get_plot_mouse_pos",
    [] (ImAxis_ x_axis, ImAxis_ y_axis) -> ImPlotPoint {
        return ImPlot::GetPlotMousePos(x_axis, y_axis);
    }, nb::arg("x_axis") = IMPLOT_AUTO, nb::arg("y_axis") = IMPLOT_AUTO);

mod_implot.def("get_plot_limits",
    [] (ImAxis_ x_axis, ImAxis_ y_axis) -> ImPlotRect {
        return ImPlot::GetPlotLimits(x_axis, y_axis);
    }, nb::arg("x_axis") = IMPLOT_AUTO, nb::arg("y_axis") = IMPLOT_AUTO);

mod_implot.def("is_plot_hovered",
    [] () -> bool {
        return ImPlot::IsPlotHovered();
    });

mod_implot.def("is_axis_hovered",
    [] (ImAxis_ axis) -> bool {
        return ImPlot::IsAxisHovered(axis);
    }, nb::arg("axis"));

mod_implot.def("is_subplots_hovered",
    [] () -> bool {
        return ImPlot::IsSubplotsHovered();
    });

mod_implot.def("is_plot_selected",
    [] () -> bool {
        return ImPlot::IsPlotSelected();
    });

mod_implot.def("get_plot_selection",
    [] (ImAxis_ x_axis, ImAxis_ y_axis) -> ImPlotRect {
        return ImPlot::GetPlotSelection(x_axis, y_axis);
    }, nb::arg("x_axis") = IMPLOT_AUTO, nb::arg("y_axis") = IMPLOT_AUTO);

mod_implot.def("cancel_plot_selection",
    [] () -> void {
        return ImPlot::CancelPlotSelection();
    });

mod_implot.def("hide_next_item",
    [] (bool hidden, ImPlotCond_ cond) -> void {
        return ImPlot::HideNextItem(hidden, cond);
    }, nb::arg("hidden") = true, nb::arg("cond") = ImPlotCond_Once);

mod_implot.def("begin_aligned_plots",
    [] (const char* group_id, bool vertical) -> bool {
        return ImPlot::BeginAlignedPlots(group_id, vertical);
    }, nb::arg("group_id"), nb::arg("vertical") = true);

mod_implot.def("end_aligned_plots",
    [] () -> void {
        return ImPlot::EndAlignedPlots();
    });

mod_implot.def("begin_legend_popup",
    [] (const char* label_id, ImGuiMouseButton mouse_button) -> bool {
        return ImPlot::BeginLegendPopup(label_id, mouse_button);
    }, nb::arg("label_id"), nb::arg("mouse_button") = 1);

mod_implot.def("end_legend_popup",
    [] () -> void {
        return ImPlot::EndLegendPopup();
    });

mod_implot.def("is_legend_entry_hovered",
    [] (const char* label_id) -> bool {
        return ImPlot::IsLegendEntryHovered(label_id);
    }, nb::arg("label_id"));

mod_implot.def("begin_drag_drop_target_plot",
    [] () -> bool {
        return ImPlot::BeginDragDropTargetPlot();
    });

mod_implot.def("begin_drag_drop_target_axis",
    [] (ImAxis_ axis) -> bool {
        return ImPlot::BeginDragDropTargetAxis(axis);
    }, nb::arg("axis"));

mod_implot.def("begin_drag_drop_target_legend",
    [] () -> bool {
        return ImPlot::BeginDragDropTargetLegend();
    });

mod_implot.def("end_drag_drop_target",
    [] () -> void {
        return ImPlot::EndDragDropTarget();
    });

mod_implot.def("begin_drag_drop_source_plot",
    [] (ImGuiDragDropFlags flags) -> bool {
        return ImPlot::BeginDragDropSourcePlot(flags);
    }, nb::arg("flags") = 0);

mod_implot.def("begin_drag_drop_source_axis",
    [] (ImAxis_ axis, ImGuiDragDropFlags flags) -> bool {
        return ImPlot::BeginDragDropSourceAxis(axis, flags);
    }, nb::arg("axis"), nb::arg("flags") = 0);

mod_implot.def("begin_drag_drop_source_item",
    [] (const char* label_id, ImGuiDragDropFlags flags) -> bool {
        return ImPlot::BeginDragDropSourceItem(label_id, flags);
    }, nb::arg("label_id"), nb::arg("flags") = 0);

mod_implot.def("end_drag_drop_source",
    [] () -> void {
        return ImPlot::EndDragDropSource();
    });

mod_implot.def("style_colors_auto",
    [] () -> void {
        return ImPlot::StyleColorsAuto();
    });

mod_implot.def("style_colors_classic",
    [] () -> void {
        return ImPlot::StyleColorsClassic();
    });

mod_implot.def("style_colors_dark",
    [] () -> void {
        return ImPlot::StyleColorsDark();
    });

mod_implot.def("style_colors_light",
    [] () -> void {
        return ImPlot::StyleColorsLight();
    });

mod_implot.def("push_style_color",
    [] (ImPlotCol_ idx, Color col) -> void {
        return ImPlot::PushStyleColor(idx, col);
    }, nb::arg("idx"), nb::arg("col"));

mod_implot.def("push_style_color_im_vec4",
    [] (ImPlotCol_ idx, ImVec4 col) -> void {
        return ImPlot::PushStyleColor(idx, col);
    }, nb::arg("idx"), nb::arg("col"));

mod_implot.def("pop_style_color",
    [] (int count) -> void {
        return ImPlot::PopStyleColor(count);
    }, nb::arg("count") = 1);

mod_implot.def("push_style_var",
    [] (ImPlotStyleVar_ idx, float val) -> void {
        return ImPlot::PushStyleVar(idx, val);
    }, nb::arg("idx"), nb::arg("val"));

mod_implot.def("push_style_var_int",
    [] (ImPlotStyleVar_ idx, int val) -> void {
        return ImPlot::PushStyleVar(idx, val);
    }, nb::arg("idx"), nb::arg("val"));

mod_implot.def("push_style_var_im_vec2",
    [] (ImPlotStyleVar_ idx, ImVec2 val) -> void {
        return ImPlot::PushStyleVar(idx, val);
    }, nb::arg("idx"), nb::arg("val"));

mod_implot.def("pop_style_var",
    [] (int count) -> void {
        return ImPlot::PopStyleVar(count);
    }, nb::arg("count") = 1);

mod_implot.def("get_last_item_color",
    [] () -> ImVec4 {
        return ImPlot::GetLastItemColor();
    });

mod_implot.def("get_style_color_name",
    [] (ImPlotCol_ idx) -> const char* {
        return ImPlot::GetStyleColorName(idx);
    }, nb::arg("idx"));

mod_implot.def("get_marker_name",
    [] (ImPlotMarker_ idx) -> const char* {
        return ImPlot::GetMarkerName(idx);
    }, nb::arg("idx"));

mod_implot.def("next_marker",
    [] () -> ImPlotMarker_ {
        return (ImPlotMarker_)ImPlot::NextMarker();
    });

mod_implot.def("add_colormap",
    [] (const char* name, const ImVec4* cols, int size, bool qual) -> ImPlotColormap_ {
        return (ImPlotColormap_)ImPlot::AddColormap(name, cols, size, qual);
    }, nb::arg("name"), nb::arg("cols"), nb::arg("size"), nb::arg("qual") = true);

mod_implot.def("get_colormap_count",
    [] () -> int {
        return ImPlot::GetColormapCount();
    });

mod_implot.def("get_colormap_name",
    [] (ImPlotColormap_ cmap) -> const char* {
        return ImPlot::GetColormapName(cmap);
    }, nb::arg("cmap"));

mod_implot.def("get_colormap_index",
    [] (const char* name) -> ImPlotColormap_ {
        return (ImPlotColormap_)ImPlot::GetColormapIndex(name);
    }, nb::arg("name"));

mod_implot.def("push_colormap",
    [] (ImPlotColormap_ cmap) -> void {
        return ImPlot::PushColormap(cmap);
    }, nb::arg("cmap"));

mod_implot.def("push_colormap_str",
    [] (const char* name) -> void {
        return ImPlot::PushColormap(name);
    }, nb::arg("name"));

mod_implot.def("pop_colormap",
    [] (int count) -> void {
        return ImPlot::PopColormap(count);
    }, nb::arg("count") = 1);

mod_implot.def("next_colormap_color",
    [] () -> ImVec4 {
        return ImPlot::NextColormapColor();
    });

mod_implot.def("get_colormap_size",
    [] (ImPlotColormap_ cmap) -> int {
        return ImPlot::GetColormapSize(cmap);
    }, nb::arg("cmap") = IMPLOT_AUTO);

mod_implot.def("get_colormap_color",
    [] (int idx, ImPlotColormap_ cmap) -> ImVec4 {
        return ImPlot::GetColormapColor(idx, cmap);
    }, nb::arg("idx"), nb::arg("cmap") = IMPLOT_AUTO);

mod_implot.def("sample_colormap",
    [] (float t, ImPlotColormap_ cmap) -> ImVec4 {
        return ImPlot::SampleColormap(t, cmap);
    }, nb::arg("t"), nb::arg("cmap") = IMPLOT_AUTO);

mod_implot.def("colormap_scale",
    [] (const char* label, float scale_min, float scale_max, ImVec2 size, const char* format, ImPlotColormapScaleFlags_ flags, ImPlotColormap_ cmap) -> void {
        return ImPlot::ColormapScale(label, scale_min, scale_max, size, format, flags, cmap);
    }, nb::arg("label"), nb::arg("scale_min"), nb::arg("scale_max"), nb::arg("size") = nb::make_tuple(0, 0), nb::arg("format") = "%g", nb::arg("flags") = 0, nb::arg("cmap") = IMPLOT_AUTO);

mod_implot.def("colormap_button",
    [] (const char* label, ImVec2 size, ImPlotColormap_ cmap) -> bool {
        return ImPlot::ColormapButton(label, size, cmap);
    }, nb::arg("label"), nb::arg("size") = nb::make_tuple(0, 0), nb::arg("cmap") = IMPLOT_AUTO);

mod_implot.def("bust_color_cache",
    [] (const char* plot_title_id) -> void {
        return ImPlot::BustColorCache(plot_title_id);
    }, nb::arg("plot_title_id") = nullptr);

mod_implot.def("item_icon",
    [] (ImVec4 col) -> void {
        return ImPlot::ItemIcon(col);
    }, nb::arg("col"));

mod_implot.def("item_icon_im_u32",
    [] (Color col) -> void {
        return ImPlot::ItemIcon(col);
    }, nb::arg("col"));

mod_implot.def("colormap_icon",
    [] (ImPlotColormap_ cmap) -> void {
        return ImPlot::ColormapIcon(cmap);
    }, nb::arg("cmap"));

mod_implot.def("get_plot_draw_list",
    [] () -> DrawList* {
        return new DrawList(ImPlot::GetPlotDrawList());
    });

mod_implot.def("push_plot_clip_rect",
    [] (float expand) -> void {
        return ImPlot::PushPlotClipRect(expand);
    }, nb::arg("expand") = 0);

mod_implot.def("pop_plot_clip_rect",
    [] () -> void {
        return ImPlot::PopPlotClipRect();
    });

mod_implot.def("show_style_selector",
    [] (const char* label) -> bool {
        return ImPlot::ShowStyleSelector(label);
    }, nb::arg("label"));

mod_implot.def("show_colormap_selector",
    [] (const char* label) -> bool {
        return ImPlot::ShowColormapSelector(label);
    }, nb::arg("label"));

mod_implot.def("show_input_map_selector",
    [] (const char* label) -> bool {
        return ImPlot::ShowInputMapSelector(label);
    }, nb::arg("label"));

mod_implot.def("show_style_editor",
    [] () -> void {
        return ImPlot::ShowStyleEditor();
    });

mod_implot.def("show_user_guide",
    [] () -> void {
        return ImPlot::ShowUserGuide();
    });

mod_implot.def("show_metrics_window",
    [] (bool p_popen) -> bool {
        ImPlot::ShowMetricsWindow(&p_popen);
        return p_popen;
    }, nb::arg("p_popen") = nullptr);

