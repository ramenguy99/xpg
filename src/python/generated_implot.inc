nb::enum_<ImAxis_>(mod_implot, "Axis", nb::is_arithmetic() )
    .value("X1", ImAxis_X1)
    .value("X2", ImAxis_X2)
    .value("X3", ImAxis_X3)
    .value("Y1", ImAxis_Y1)
    .value("Y2", ImAxis_Y2)
    .value("Y3", ImAxis_Y3)
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
    .value("LINE", ImPlotCol_Line)
    .value("FILL", ImPlotCol_Fill)
    .value("MARKER_OUTLINE", ImPlotCol_MarkerOutline)
    .value("MARKER_FILL", ImPlotCol_MarkerFill)
    .value("ERROR_BAR", ImPlotCol_ErrorBar)
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
    .value("LINE_WEIGHT", ImPlotStyleVar_LineWeight)
    .value("MARKER", ImPlotStyleVar_Marker)
    .value("MARKER_SIZE", ImPlotStyleVar_MarkerSize)
    .value("MARKER_WEIGHT", ImPlotStyleVar_MarkerWeight)
    .value("FILL_ALPHA", ImPlotStyleVar_FillAlpha)
    .value("ERROR_BAR_SIZE", ImPlotStyleVar_ErrorBarSize)
    .value("ERROR_BAR_WEIGHT", ImPlotStyleVar_ErrorBarWeight)
    .value("DIGITAL_BIT_HEIGHT", ImPlotStyleVar_DigitalBitHeight)
    .value("DIGITAL_BIT_GAP", ImPlotStyleVar_DigitalBitGap)
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
    .value("PLOT_DEFAULT_SIZE", ImPlotStyleVar_PlotDefaultSize)
    .value("PLOT_MIN_SIZE", ImPlotStyleVar_PlotMinSize)
;


nb::enum_<ImPlotScale_>(mod_implot, "Scale", nb::is_arithmetic() )
    .value("LINEAR", ImPlotScale_Linear)
    .value("TIME", ImPlotScale_Time)
    .value("LOG10", ImPlotScale_Log10)
    .value("SYM_LOG", ImPlotScale_SymLog)
;


nb::enum_<ImPlotMarker_>(mod_implot, "Marker", nb::is_arithmetic() )
    .value("NONE", ImPlotMarker_None)
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
    .def_rw("line_weight", &ImPlotStyle::LineWeight)
    .def_rw("marker", &ImPlotStyle::Marker)
    .def_rw("marker_size", &ImPlotStyle::MarkerSize)
    .def_rw("marker_weight", &ImPlotStyle::MarkerWeight)
    .def_rw("fill_alpha", &ImPlotStyle::FillAlpha)
    .def_rw("error_bar_size", &ImPlotStyle::ErrorBarSize)
    .def_rw("error_bar_weight", &ImPlotStyle::ErrorBarWeight)
    .def_rw("digital_bit_height", &ImPlotStyle::DigitalBitHeight)
    .def_rw("digital_bit_gap", &ImPlotStyle::DigitalBitGap)
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
    .def_rw("plot_default_size", &ImPlotStyle::PlotDefaultSize)
    .def_rw("plot_min_size", &ImPlotStyle::PlotMinSize)
    .def_prop_rw("color_line", [](ImPlotStyle& s) -> ImVec4 { return s.Colors[ImPlotCol_Line]; }, [](ImPlotStyle& s, ImVec4 value){ s.Colors[ImPlotCol_Line] = value; })
    .def_prop_rw("color_fill", [](ImPlotStyle& s) -> ImVec4 { return s.Colors[ImPlotCol_Fill]; }, [](ImPlotStyle& s, ImVec4 value){ s.Colors[ImPlotCol_Fill] = value; })
    .def_prop_rw("color_marker_outline", [](ImPlotStyle& s) -> ImVec4 { return s.Colors[ImPlotCol_MarkerOutline]; }, [](ImPlotStyle& s, ImVec4 value){ s.Colors[ImPlotCol_MarkerOutline] = value; })
    .def_prop_rw("color_marker_fill", [](ImPlotStyle& s) -> ImVec4 { return s.Colors[ImPlotCol_MarkerFill]; }, [](ImPlotStyle& s, ImVec4 value){ s.Colors[ImPlotCol_MarkerFill] = value; })
    .def_prop_rw("color_error_bar", [](ImPlotStyle& s) -> ImVec4 { return s.Colors[ImPlotCol_ErrorBar]; }, [](ImPlotStyle& s, ImVec4 value){ s.Colors[ImPlotCol_ErrorBar] = value; })
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

