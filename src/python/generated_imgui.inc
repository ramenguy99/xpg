nb::enum_<ImGuiWindowFlags_>(mod_imgui, "WindowFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImGuiWindowFlags_None)
    .value("NO_TITLE_BAR", ImGuiWindowFlags_NoTitleBar)
    .value("NO_RESIZE", ImGuiWindowFlags_NoResize)
    .value("NO_MOVE", ImGuiWindowFlags_NoMove)
    .value("NO_SCROLLBAR", ImGuiWindowFlags_NoScrollbar)
    .value("NO_SCROLL_WITH_MOUSE", ImGuiWindowFlags_NoScrollWithMouse)
    .value("NO_COLLAPSE", ImGuiWindowFlags_NoCollapse)
    .value("ALWAYS_AUTO_RESIZE", ImGuiWindowFlags_AlwaysAutoResize)
    .value("NO_BACKGROUND", ImGuiWindowFlags_NoBackground)
    .value("NO_SAVED_SETTINGS", ImGuiWindowFlags_NoSavedSettings)
    .value("NO_MOUSE_INPUTS", ImGuiWindowFlags_NoMouseInputs)
    .value("MENU_BAR", ImGuiWindowFlags_MenuBar)
    .value("HORIZONTAL_SCROLLBAR", ImGuiWindowFlags_HorizontalScrollbar)
    .value("NO_FOCUS_ON_APPEARING", ImGuiWindowFlags_NoFocusOnAppearing)
    .value("NO_BRING_TO_FRONT_ON_FOCUS", ImGuiWindowFlags_NoBringToFrontOnFocus)
    .value("ALWAYS_VERTICAL_SCROLLBAR", ImGuiWindowFlags_AlwaysVerticalScrollbar)
    .value("ALWAYS_HORIZONTAL_SCROLLBAR", ImGuiWindowFlags_AlwaysHorizontalScrollbar)
    .value("NO_NAV_INPUTS", ImGuiWindowFlags_NoNavInputs)
    .value("NO_NAV_FOCUS", ImGuiWindowFlags_NoNavFocus)
    .value("UNSAVED_DOCUMENT", ImGuiWindowFlags_UnsavedDocument)
    .value("NO_DOCKING", ImGuiWindowFlags_NoDocking)
    .value("NO_NAV", ImGuiWindowFlags_NoNav)
    .value("NO_DECORATION", ImGuiWindowFlags_NoDecoration)
    .value("NO_INPUTS", ImGuiWindowFlags_NoInputs)
    .value("DOCK_NODE_HOST", ImGuiWindowFlags_DockNodeHost)
    .value("CHILD_WINDOW", ImGuiWindowFlags_ChildWindow)
    .value("TOOLTIP", ImGuiWindowFlags_Tooltip)
    .value("POPUP", ImGuiWindowFlags_Popup)
    .value("MODAL", ImGuiWindowFlags_Modal)
    .value("CHILD_MENU", ImGuiWindowFlags_ChildMenu)
    .value("NAV_FLATTENED", ImGuiWindowFlags_NavFlattened)
    .value("ALWAYS_USE_WINDOW_PADDING", ImGuiWindowFlags_AlwaysUseWindowPadding)
    ;

nb::enum_<ImGuiChildFlags_>(mod_imgui, "ChildFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImGuiChildFlags_None)
    .value("BORDERS", ImGuiChildFlags_Borders)
    .value("ALWAYS_USE_WINDOW_PADDING", ImGuiChildFlags_AlwaysUseWindowPadding)
    .value("RESIZE_X", ImGuiChildFlags_ResizeX)
    .value("RESIZE_Y", ImGuiChildFlags_ResizeY)
    .value("AUTO_RESIZE_X", ImGuiChildFlags_AutoResizeX)
    .value("AUTO_RESIZE_Y", ImGuiChildFlags_AutoResizeY)
    .value("ALWAYS_AUTO_RESIZE", ImGuiChildFlags_AlwaysAutoResize)
    .value("FRAME_STYLE", ImGuiChildFlags_FrameStyle)
    .value("NAV_FLATTENED", ImGuiChildFlags_NavFlattened)
    .value("BORDER", ImGuiChildFlags_Border)
    ;

nb::enum_<ImGuiItemFlags_>(mod_imgui, "ItemFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImGuiItemFlags_None)
    .value("NO_TAB_STOP", ImGuiItemFlags_NoTabStop)
    .value("NO_NAV", ImGuiItemFlags_NoNav)
    .value("NO_NAV_DEFAULT_FOCUS", ImGuiItemFlags_NoNavDefaultFocus)
    .value("BUTTON_REPEAT", ImGuiItemFlags_ButtonRepeat)
    .value("AUTO_CLOSE_POPUPS", ImGuiItemFlags_AutoClosePopups)
    .value("ALLOW_DUPLICATE_ID", ImGuiItemFlags_AllowDuplicateId)
    ;

nb::enum_<ImGuiInputTextFlags_>(mod_imgui, "InputTextFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImGuiInputTextFlags_None)
    .value("CHARS_DECIMAL", ImGuiInputTextFlags_CharsDecimal)
    .value("CHARS_HEXADECIMAL", ImGuiInputTextFlags_CharsHexadecimal)
    .value("CHARS_SCIENTIFIC", ImGuiInputTextFlags_CharsScientific)
    .value("CHARS_UPPERCASE", ImGuiInputTextFlags_CharsUppercase)
    .value("CHARS_NO_BLANK", ImGuiInputTextFlags_CharsNoBlank)
    .value("ALLOW_TAB_INPUT", ImGuiInputTextFlags_AllowTabInput)
    .value("ENTER_RETURNS_TRUE", ImGuiInputTextFlags_EnterReturnsTrue)
    .value("ESCAPE_CLEARS_ALL", ImGuiInputTextFlags_EscapeClearsAll)
    .value("CTRL_ENTER_FOR_NEW_LINE", ImGuiInputTextFlags_CtrlEnterForNewLine)
    .value("READ_ONLY", ImGuiInputTextFlags_ReadOnly)
    .value("PASSWORD", ImGuiInputTextFlags_Password)
    .value("ALWAYS_OVERWRITE", ImGuiInputTextFlags_AlwaysOverwrite)
    .value("AUTO_SELECT_ALL", ImGuiInputTextFlags_AutoSelectAll)
    .value("PARSE_EMPTY_REF_VAL", ImGuiInputTextFlags_ParseEmptyRefVal)
    .value("DISPLAY_EMPTY_REF_VAL", ImGuiInputTextFlags_DisplayEmptyRefVal)
    .value("NO_HORIZONTAL_SCROLL", ImGuiInputTextFlags_NoHorizontalScroll)
    .value("NO_UNDO_REDO", ImGuiInputTextFlags_NoUndoRedo)
    .value("ELIDE_LEFT", ImGuiInputTextFlags_ElideLeft)
    .value("CALLBACK_COMPLETION", ImGuiInputTextFlags_CallbackCompletion)
    .value("CALLBACK_HISTORY", ImGuiInputTextFlags_CallbackHistory)
    .value("CALLBACK_ALWAYS", ImGuiInputTextFlags_CallbackAlways)
    .value("CALLBACK_CHAR_FILTER", ImGuiInputTextFlags_CallbackCharFilter)
    .value("CALLBACK_RESIZE", ImGuiInputTextFlags_CallbackResize)
    .value("CALLBACK_EDIT", ImGuiInputTextFlags_CallbackEdit)
    ;

nb::enum_<ImGuiTreeNodeFlags_>(mod_imgui, "TreeNodeFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImGuiTreeNodeFlags_None)
    .value("SELECTED", ImGuiTreeNodeFlags_Selected)
    .value("FRAMED", ImGuiTreeNodeFlags_Framed)
    .value("ALLOW_OVERLAP", ImGuiTreeNodeFlags_AllowOverlap)
    .value("NO_TREE_PUSH_ON_OPEN", ImGuiTreeNodeFlags_NoTreePushOnOpen)
    .value("NO_AUTO_OPEN_ON_LOG", ImGuiTreeNodeFlags_NoAutoOpenOnLog)
    .value("DEFAULT_OPEN", ImGuiTreeNodeFlags_DefaultOpen)
    .value("OPEN_ON_DOUBLE_CLICK", ImGuiTreeNodeFlags_OpenOnDoubleClick)
    .value("OPEN_ON_ARROW", ImGuiTreeNodeFlags_OpenOnArrow)
    .value("LEAF", ImGuiTreeNodeFlags_Leaf)
    .value("BULLET", ImGuiTreeNodeFlags_Bullet)
    .value("FRAME_PADDING", ImGuiTreeNodeFlags_FramePadding)
    .value("SPAN_AVAIL_WIDTH", ImGuiTreeNodeFlags_SpanAvailWidth)
    .value("SPAN_FULL_WIDTH", ImGuiTreeNodeFlags_SpanFullWidth)
    .value("SPAN_LABEL_WIDTH", ImGuiTreeNodeFlags_SpanLabelWidth)
    .value("SPAN_ALL_COLUMNS", ImGuiTreeNodeFlags_SpanAllColumns)
    .value("LABEL_SPAN_ALL_COLUMNS", ImGuiTreeNodeFlags_LabelSpanAllColumns)
    .value("NAV_LEFT_JUMPS_BACK_HERE", ImGuiTreeNodeFlags_NavLeftJumpsBackHere)
    .value("COLLAPSING_HEADER", ImGuiTreeNodeFlags_CollapsingHeader)
    .value("ALLOW_ITEM_OVERLAP", ImGuiTreeNodeFlags_AllowItemOverlap)
    .value("SPAN_TEXT_WIDTH", ImGuiTreeNodeFlags_SpanTextWidth)
    ;

nb::enum_<ImGuiPopupFlags_>(mod_imgui, "PopupFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImGuiPopupFlags_None)
    .value("MOUSE_BUTTON_LEFT", ImGuiPopupFlags_MouseButtonLeft)
    .value("MOUSE_BUTTON_RIGHT", ImGuiPopupFlags_MouseButtonRight)
    .value("MOUSE_BUTTON_MIDDLE", ImGuiPopupFlags_MouseButtonMiddle)
    .value("MOUSE_BUTTON_MASK_", ImGuiPopupFlags_MouseButtonMask_)
    .value("MOUSE_BUTTON_DEFAULT_", ImGuiPopupFlags_MouseButtonDefault_)
    .value("NO_REOPEN", ImGuiPopupFlags_NoReopen)
    .value("NO_OPEN_OVER_EXISTING_POPUP", ImGuiPopupFlags_NoOpenOverExistingPopup)
    .value("NO_OPEN_OVER_ITEMS", ImGuiPopupFlags_NoOpenOverItems)
    .value("ANY_POPUP_ID", ImGuiPopupFlags_AnyPopupId)
    .value("ANY_POPUP_LEVEL", ImGuiPopupFlags_AnyPopupLevel)
    .value("ANY_POPUP", ImGuiPopupFlags_AnyPopup)
    ;

nb::enum_<ImGuiSelectableFlags_>(mod_imgui, "SelectableFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImGuiSelectableFlags_None)
    .value("NO_AUTO_CLOSE_POPUPS", ImGuiSelectableFlags_NoAutoClosePopups)
    .value("SPAN_ALL_COLUMNS", ImGuiSelectableFlags_SpanAllColumns)
    .value("ALLOW_DOUBLE_CLICK", ImGuiSelectableFlags_AllowDoubleClick)
    .value("DISABLED", ImGuiSelectableFlags_Disabled)
    .value("ALLOW_OVERLAP", ImGuiSelectableFlags_AllowOverlap)
    .value("HIGHLIGHT", ImGuiSelectableFlags_Highlight)
    .value("DONT_CLOSE_POPUPS", ImGuiSelectableFlags_DontClosePopups)
    .value("ALLOW_ITEM_OVERLAP", ImGuiSelectableFlags_AllowItemOverlap)
    ;

nb::enum_<ImGuiComboFlags_>(mod_imgui, "ComboFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImGuiComboFlags_None)
    .value("POPUP_ALIGN_LEFT", ImGuiComboFlags_PopupAlignLeft)
    .value("HEIGHT_SMALL", ImGuiComboFlags_HeightSmall)
    .value("HEIGHT_REGULAR", ImGuiComboFlags_HeightRegular)
    .value("HEIGHT_LARGE", ImGuiComboFlags_HeightLarge)
    .value("HEIGHT_LARGEST", ImGuiComboFlags_HeightLargest)
    .value("NO_ARROW_BUTTON", ImGuiComboFlags_NoArrowButton)
    .value("NO_PREVIEW", ImGuiComboFlags_NoPreview)
    .value("WIDTH_FIT_PREVIEW", ImGuiComboFlags_WidthFitPreview)
    .value("HEIGHT_MASK_", ImGuiComboFlags_HeightMask_)
    ;

nb::enum_<ImGuiTabBarFlags_>(mod_imgui, "TabBarFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImGuiTabBarFlags_None)
    .value("REORDERABLE", ImGuiTabBarFlags_Reorderable)
    .value("AUTO_SELECT_NEW_TABS", ImGuiTabBarFlags_AutoSelectNewTabs)
    .value("TAB_LIST_POPUP_BUTTON", ImGuiTabBarFlags_TabListPopupButton)
    .value("NO_CLOSE_WITH_MIDDLE_MOUSE_BUTTON", ImGuiTabBarFlags_NoCloseWithMiddleMouseButton)
    .value("NO_TAB_LIST_SCROLLING_BUTTONS", ImGuiTabBarFlags_NoTabListScrollingButtons)
    .value("NO_TOOLTIP", ImGuiTabBarFlags_NoTooltip)
    .value("DRAW_SELECTED_OVERLINE", ImGuiTabBarFlags_DrawSelectedOverline)
    .value("FITTING_POLICY_RESIZE_DOWN", ImGuiTabBarFlags_FittingPolicyResizeDown)
    .value("FITTING_POLICY_SCROLL", ImGuiTabBarFlags_FittingPolicyScroll)
    .value("FITTING_POLICY_MASK_", ImGuiTabBarFlags_FittingPolicyMask_)
    .value("FITTING_POLICY_DEFAULT_", ImGuiTabBarFlags_FittingPolicyDefault_)
    ;

nb::enum_<ImGuiTabItemFlags_>(mod_imgui, "TabItemFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImGuiTabItemFlags_None)
    .value("UNSAVED_DOCUMENT", ImGuiTabItemFlags_UnsavedDocument)
    .value("SET_SELECTED", ImGuiTabItemFlags_SetSelected)
    .value("NO_CLOSE_WITH_MIDDLE_MOUSE_BUTTON", ImGuiTabItemFlags_NoCloseWithMiddleMouseButton)
    .value("NO_PUSH_ID", ImGuiTabItemFlags_NoPushId)
    .value("NO_TOOLTIP", ImGuiTabItemFlags_NoTooltip)
    .value("NO_REORDER", ImGuiTabItemFlags_NoReorder)
    .value("LEADING", ImGuiTabItemFlags_Leading)
    .value("TRAILING", ImGuiTabItemFlags_Trailing)
    .value("NO_ASSUMED_CLOSURE", ImGuiTabItemFlags_NoAssumedClosure)
    ;

nb::enum_<ImGuiFocusedFlags_>(mod_imgui, "FocusedFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImGuiFocusedFlags_None)
    .value("CHILD_WINDOWS", ImGuiFocusedFlags_ChildWindows)
    .value("ROOT_WINDOW", ImGuiFocusedFlags_RootWindow)
    .value("ANY_WINDOW", ImGuiFocusedFlags_AnyWindow)
    .value("NO_POPUP_HIERARCHY", ImGuiFocusedFlags_NoPopupHierarchy)
    .value("DOCK_HIERARCHY", ImGuiFocusedFlags_DockHierarchy)
    .value("ROOT_AND_CHILD_WINDOWS", ImGuiFocusedFlags_RootAndChildWindows)
    ;

nb::enum_<ImGuiHoveredFlags_>(mod_imgui, "HoveredFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImGuiHoveredFlags_None)
    .value("CHILD_WINDOWS", ImGuiHoveredFlags_ChildWindows)
    .value("ROOT_WINDOW", ImGuiHoveredFlags_RootWindow)
    .value("ANY_WINDOW", ImGuiHoveredFlags_AnyWindow)
    .value("NO_POPUP_HIERARCHY", ImGuiHoveredFlags_NoPopupHierarchy)
    .value("DOCK_HIERARCHY", ImGuiHoveredFlags_DockHierarchy)
    .value("ALLOW_WHEN_BLOCKED_BY_POPUP", ImGuiHoveredFlags_AllowWhenBlockedByPopup)
    .value("ALLOW_WHEN_BLOCKED_BY_ACTIVE_ITEM", ImGuiHoveredFlags_AllowWhenBlockedByActiveItem)
    .value("ALLOW_WHEN_OVERLAPPED_BY_ITEM", ImGuiHoveredFlags_AllowWhenOverlappedByItem)
    .value("ALLOW_WHEN_OVERLAPPED_BY_WINDOW", ImGuiHoveredFlags_AllowWhenOverlappedByWindow)
    .value("ALLOW_WHEN_DISABLED", ImGuiHoveredFlags_AllowWhenDisabled)
    .value("NO_NAV_OVERRIDE", ImGuiHoveredFlags_NoNavOverride)
    .value("ALLOW_WHEN_OVERLAPPED", ImGuiHoveredFlags_AllowWhenOverlapped)
    .value("RECT_ONLY", ImGuiHoveredFlags_RectOnly)
    .value("ROOT_AND_CHILD_WINDOWS", ImGuiHoveredFlags_RootAndChildWindows)
    .value("FOR_TOOLTIP", ImGuiHoveredFlags_ForTooltip)
    .value("STATIONARY", ImGuiHoveredFlags_Stationary)
    .value("DELAY_NONE", ImGuiHoveredFlags_DelayNone)
    .value("DELAY_SHORT", ImGuiHoveredFlags_DelayShort)
    .value("DELAY_NORMAL", ImGuiHoveredFlags_DelayNormal)
    .value("NO_SHARED_DELAY", ImGuiHoveredFlags_NoSharedDelay)
    ;

nb::enum_<ImGuiDockNodeFlags_>(mod_imgui, "DockNodeFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImGuiDockNodeFlags_None)
    .value("KEEP_ALIVE_ONLY", ImGuiDockNodeFlags_KeepAliveOnly)
    .value("NO_DOCKING_OVER_CENTRAL_NODE", ImGuiDockNodeFlags_NoDockingOverCentralNode)
    .value("PASSTHRU_CENTRAL_NODE", ImGuiDockNodeFlags_PassthruCentralNode)
    .value("NO_DOCKING_SPLIT", ImGuiDockNodeFlags_NoDockingSplit)
    .value("NO_RESIZE", ImGuiDockNodeFlags_NoResize)
    .value("AUTO_HIDE_TAB_BAR", ImGuiDockNodeFlags_AutoHideTabBar)
    .value("NO_UNDOCKING", ImGuiDockNodeFlags_NoUndocking)
    .value("NO_SPLIT", ImGuiDockNodeFlags_NoSplit)
    .value("NO_DOCKING_IN_CENTRAL_NODE", ImGuiDockNodeFlags_NoDockingInCentralNode)
    ;

nb::enum_<ImGuiDragDropFlags_>(mod_imgui, "DragDropFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImGuiDragDropFlags_None)
    .value("SOURCE_NO_PREVIEW_TOOLTIP", ImGuiDragDropFlags_SourceNoPreviewTooltip)
    .value("SOURCE_NO_DISABLE_HOVER", ImGuiDragDropFlags_SourceNoDisableHover)
    .value("SOURCE_NO_HOLD_TO_OPEN_OTHERS", ImGuiDragDropFlags_SourceNoHoldToOpenOthers)
    .value("SOURCE_ALLOW_NULL_ID", ImGuiDragDropFlags_SourceAllowNullID)
    .value("SOURCE_EXTERN", ImGuiDragDropFlags_SourceExtern)
    .value("PAYLOAD_AUTO_EXPIRE", ImGuiDragDropFlags_PayloadAutoExpire)
    .value("PAYLOAD_NO_CROSS_CONTEXT", ImGuiDragDropFlags_PayloadNoCrossContext)
    .value("PAYLOAD_NO_CROSS_PROCESS", ImGuiDragDropFlags_PayloadNoCrossProcess)
    .value("ACCEPT_BEFORE_DELIVERY", ImGuiDragDropFlags_AcceptBeforeDelivery)
    .value("ACCEPT_NO_DRAW_DEFAULT_RECT", ImGuiDragDropFlags_AcceptNoDrawDefaultRect)
    .value("ACCEPT_NO_PREVIEW_TOOLTIP", ImGuiDragDropFlags_AcceptNoPreviewTooltip)
    .value("ACCEPT_PEEK_ONLY", ImGuiDragDropFlags_AcceptPeekOnly)
    .value("SOURCE_AUTO_EXPIRE_PAYLOAD", ImGuiDragDropFlags_SourceAutoExpirePayload)
    ;

nb::enum_<ImGuiDataType_>(mod_imgui, "DataType", nb::is_arithmetic() )
    .value("S8", ImGuiDataType_S8)
    .value("U8", ImGuiDataType_U8)
    .value("S16", ImGuiDataType_S16)
    .value("U16", ImGuiDataType_U16)
    .value("S32", ImGuiDataType_S32)
    .value("U32", ImGuiDataType_U32)
    .value("S64", ImGuiDataType_S64)
    .value("U64", ImGuiDataType_U64)
    .value("FLOAT", ImGuiDataType_Float)
    .value("DOUBLE", ImGuiDataType_Double)
    .value("BOOL", ImGuiDataType_Bool)
    .value("STRING", ImGuiDataType_String)
    ;

nb::enum_<ImGuiDir>(mod_imgui, "Dir", nb::is_arithmetic() )
    .value("NONE", ImGuiDir_None)
    .value("LEFT", ImGuiDir_Left)
    .value("RIGHT", ImGuiDir_Right)
    .value("UP", ImGuiDir_Up)
    .value("DOWN", ImGuiDir_Down)
    ;

nb::enum_<ImGuiSortDirection>(mod_imgui, "SortDirection", nb::is_arithmetic() )
    .value("NONE", ImGuiSortDirection_None)
    .value("ASCENDING", ImGuiSortDirection_Ascending)
    .value("DESCENDING", ImGuiSortDirection_Descending)
    ;

nb::enum_<ImGuiKey>(mod_imgui, "Key", nb::is_arithmetic() )
    .value("NONE", ImGuiKey_None)
    .value("NAMED_KEY__B_E_G_I_N", ImGuiKey_NamedKey_BEGIN)
    .value("TAB", ImGuiKey_Tab)
    .value("LEFT_ARROW", ImGuiKey_LeftArrow)
    .value("RIGHT_ARROW", ImGuiKey_RightArrow)
    .value("UP_ARROW", ImGuiKey_UpArrow)
    .value("DOWN_ARROW", ImGuiKey_DownArrow)
    .value("PAGE_UP", ImGuiKey_PageUp)
    .value("PAGE_DOWN", ImGuiKey_PageDown)
    .value("HOME", ImGuiKey_Home)
    .value("END", ImGuiKey_End)
    .value("INSERT", ImGuiKey_Insert)
    .value("DELETE", ImGuiKey_Delete)
    .value("BACKSPACE", ImGuiKey_Backspace)
    .value("SPACE", ImGuiKey_Space)
    .value("ENTER", ImGuiKey_Enter)
    .value("ESCAPE", ImGuiKey_Escape)
    .value("LEFT_CTRL", ImGuiKey_LeftCtrl)
    .value("LEFT_SHIFT", ImGuiKey_LeftShift)
    .value("LEFT_ALT", ImGuiKey_LeftAlt)
    .value("LEFT_SUPER", ImGuiKey_LeftSuper)
    .value("RIGHT_CTRL", ImGuiKey_RightCtrl)
    .value("RIGHT_SHIFT", ImGuiKey_RightShift)
    .value("RIGHT_ALT", ImGuiKey_RightAlt)
    .value("RIGHT_SUPER", ImGuiKey_RightSuper)
    .value("MENU", ImGuiKey_Menu)
    .value("KEY_0", ImGuiKey_0)
    .value("KEY_1", ImGuiKey_1)
    .value("KEY_2", ImGuiKey_2)
    .value("KEY_3", ImGuiKey_3)
    .value("KEY_4", ImGuiKey_4)
    .value("KEY_5", ImGuiKey_5)
    .value("KEY_6", ImGuiKey_6)
    .value("KEY_7", ImGuiKey_7)
    .value("KEY_8", ImGuiKey_8)
    .value("KEY_9", ImGuiKey_9)
    .value("A", ImGuiKey_A)
    .value("B", ImGuiKey_B)
    .value("C", ImGuiKey_C)
    .value("D", ImGuiKey_D)
    .value("E", ImGuiKey_E)
    .value("F", ImGuiKey_F)
    .value("G", ImGuiKey_G)
    .value("H", ImGuiKey_H)
    .value("I", ImGuiKey_I)
    .value("J", ImGuiKey_J)
    .value("K", ImGuiKey_K)
    .value("L", ImGuiKey_L)
    .value("M", ImGuiKey_M)
    .value("N", ImGuiKey_N)
    .value("O", ImGuiKey_O)
    .value("P", ImGuiKey_P)
    .value("Q", ImGuiKey_Q)
    .value("R", ImGuiKey_R)
    .value("S", ImGuiKey_S)
    .value("T", ImGuiKey_T)
    .value("U", ImGuiKey_U)
    .value("V", ImGuiKey_V)
    .value("W", ImGuiKey_W)
    .value("X", ImGuiKey_X)
    .value("Y", ImGuiKey_Y)
    .value("Z", ImGuiKey_Z)
    .value("F1", ImGuiKey_F1)
    .value("F2", ImGuiKey_F2)
    .value("F3", ImGuiKey_F3)
    .value("F4", ImGuiKey_F4)
    .value("F5", ImGuiKey_F5)
    .value("F6", ImGuiKey_F6)
    .value("F7", ImGuiKey_F7)
    .value("F8", ImGuiKey_F8)
    .value("F9", ImGuiKey_F9)
    .value("F10", ImGuiKey_F10)
    .value("F11", ImGuiKey_F11)
    .value("F12", ImGuiKey_F12)
    .value("F13", ImGuiKey_F13)
    .value("F14", ImGuiKey_F14)
    .value("F15", ImGuiKey_F15)
    .value("F16", ImGuiKey_F16)
    .value("F17", ImGuiKey_F17)
    .value("F18", ImGuiKey_F18)
    .value("F19", ImGuiKey_F19)
    .value("F20", ImGuiKey_F20)
    .value("F21", ImGuiKey_F21)
    .value("F22", ImGuiKey_F22)
    .value("F23", ImGuiKey_F23)
    .value("F24", ImGuiKey_F24)
    .value("APOSTROPHE", ImGuiKey_Apostrophe)
    .value("COMMA", ImGuiKey_Comma)
    .value("MINUS", ImGuiKey_Minus)
    .value("PERIOD", ImGuiKey_Period)
    .value("SLASH", ImGuiKey_Slash)
    .value("SEMICOLON", ImGuiKey_Semicolon)
    .value("EQUAL", ImGuiKey_Equal)
    .value("LEFT_BRACKET", ImGuiKey_LeftBracket)
    .value("BACKSLASH", ImGuiKey_Backslash)
    .value("RIGHT_BRACKET", ImGuiKey_RightBracket)
    .value("GRAVE_ACCENT", ImGuiKey_GraveAccent)
    .value("CAPS_LOCK", ImGuiKey_CapsLock)
    .value("SCROLL_LOCK", ImGuiKey_ScrollLock)
    .value("NUM_LOCK", ImGuiKey_NumLock)
    .value("PRINT_SCREEN", ImGuiKey_PrintScreen)
    .value("PAUSE", ImGuiKey_Pause)
    .value("KEYPAD0", ImGuiKey_Keypad0)
    .value("KEYPAD1", ImGuiKey_Keypad1)
    .value("KEYPAD2", ImGuiKey_Keypad2)
    .value("KEYPAD3", ImGuiKey_Keypad3)
    .value("KEYPAD4", ImGuiKey_Keypad4)
    .value("KEYPAD5", ImGuiKey_Keypad5)
    .value("KEYPAD6", ImGuiKey_Keypad6)
    .value("KEYPAD7", ImGuiKey_Keypad7)
    .value("KEYPAD8", ImGuiKey_Keypad8)
    .value("KEYPAD9", ImGuiKey_Keypad9)
    .value("KEYPAD_DECIMAL", ImGuiKey_KeypadDecimal)
    .value("KEYPAD_DIVIDE", ImGuiKey_KeypadDivide)
    .value("KEYPAD_MULTIPLY", ImGuiKey_KeypadMultiply)
    .value("KEYPAD_SUBTRACT", ImGuiKey_KeypadSubtract)
    .value("KEYPAD_ADD", ImGuiKey_KeypadAdd)
    .value("KEYPAD_ENTER", ImGuiKey_KeypadEnter)
    .value("KEYPAD_EQUAL", ImGuiKey_KeypadEqual)
    .value("APP_BACK", ImGuiKey_AppBack)
    .value("APP_FORWARD", ImGuiKey_AppForward)
    .value("OEM102", ImGuiKey_Oem102)
    .value("GAMEPAD_START", ImGuiKey_GamepadStart)
    .value("GAMEPAD_BACK", ImGuiKey_GamepadBack)
    .value("GAMEPAD_FACE_LEFT", ImGuiKey_GamepadFaceLeft)
    .value("GAMEPAD_FACE_RIGHT", ImGuiKey_GamepadFaceRight)
    .value("GAMEPAD_FACE_UP", ImGuiKey_GamepadFaceUp)
    .value("GAMEPAD_FACE_DOWN", ImGuiKey_GamepadFaceDown)
    .value("GAMEPAD_DPAD_LEFT", ImGuiKey_GamepadDpadLeft)
    .value("GAMEPAD_DPAD_RIGHT", ImGuiKey_GamepadDpadRight)
    .value("GAMEPAD_DPAD_UP", ImGuiKey_GamepadDpadUp)
    .value("GAMEPAD_DPAD_DOWN", ImGuiKey_GamepadDpadDown)
    .value("GAMEPAD_L1", ImGuiKey_GamepadL1)
    .value("GAMEPAD_R1", ImGuiKey_GamepadR1)
    .value("GAMEPAD_L2", ImGuiKey_GamepadL2)
    .value("GAMEPAD_R2", ImGuiKey_GamepadR2)
    .value("GAMEPAD_L3", ImGuiKey_GamepadL3)
    .value("GAMEPAD_R3", ImGuiKey_GamepadR3)
    .value("GAMEPAD_L_STICK_LEFT", ImGuiKey_GamepadLStickLeft)
    .value("GAMEPAD_L_STICK_RIGHT", ImGuiKey_GamepadLStickRight)
    .value("GAMEPAD_L_STICK_UP", ImGuiKey_GamepadLStickUp)
    .value("GAMEPAD_L_STICK_DOWN", ImGuiKey_GamepadLStickDown)
    .value("GAMEPAD_R_STICK_LEFT", ImGuiKey_GamepadRStickLeft)
    .value("GAMEPAD_R_STICK_RIGHT", ImGuiKey_GamepadRStickRight)
    .value("GAMEPAD_R_STICK_UP", ImGuiKey_GamepadRStickUp)
    .value("GAMEPAD_R_STICK_DOWN", ImGuiKey_GamepadRStickDown)
    .value("MOUSE_LEFT", ImGuiKey_MouseLeft)
    .value("MOUSE_RIGHT", ImGuiKey_MouseRight)
    .value("MOUSE_MIDDLE", ImGuiKey_MouseMiddle)
    .value("MOUSE_X1", ImGuiKey_MouseX1)
    .value("MOUSE_X2", ImGuiKey_MouseX2)
    .value("MOUSE_WHEEL_X", ImGuiKey_MouseWheelX)
    .value("MOUSE_WHEEL_Y", ImGuiKey_MouseWheelY)
    .value("RESERVED_FOR_MOD_CTRL", ImGuiKey_ReservedForModCtrl)
    .value("RESERVED_FOR_MOD_SHIFT", ImGuiKey_ReservedForModShift)
    .value("RESERVED_FOR_MOD_ALT", ImGuiKey_ReservedForModAlt)
    .value("RESERVED_FOR_MOD_SUPER", ImGuiKey_ReservedForModSuper)
    .value("NAMED_KEY__E_N_D", ImGuiKey_NamedKey_END)
    .value("MOD_CTRL", ImGuiKey_ModCtrl)
    .value("MOD_SHIFT", ImGuiKey_ModShift)
    .value("MOD_ALT", ImGuiKey_ModAlt)
    .value("MOD_SUPER", ImGuiKey_ModSuper)
    ;

nb::enum_<ImGuiInputFlags_>(mod_imgui, "InputFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImGuiInputFlags_None)
    .value("REPEAT", ImGuiInputFlags_Repeat)
    .value("ROUTE_ACTIVE", ImGuiInputFlags_RouteActive)
    .value("ROUTE_FOCUSED", ImGuiInputFlags_RouteFocused)
    .value("ROUTE_GLOBAL", ImGuiInputFlags_RouteGlobal)
    .value("ROUTE_ALWAYS", ImGuiInputFlags_RouteAlways)
    .value("ROUTE_OVER_FOCUSED", ImGuiInputFlags_RouteOverFocused)
    .value("ROUTE_OVER_ACTIVE", ImGuiInputFlags_RouteOverActive)
    .value("ROUTE_UNLESS_BG_FOCUSED", ImGuiInputFlags_RouteUnlessBgFocused)
    .value("ROUTE_FROM_ROOT_WINDOW", ImGuiInputFlags_RouteFromRootWindow)
    .value("TOOLTIP", ImGuiInputFlags_Tooltip)
    ;

nb::enum_<ImGuiConfigFlags_>(mod_imgui, "ConfigFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImGuiConfigFlags_None)
    .value("NAV_ENABLE_KEYBOARD", ImGuiConfigFlags_NavEnableKeyboard)
    .value("NAV_ENABLE_GAMEPAD", ImGuiConfigFlags_NavEnableGamepad)
    .value("NO_MOUSE", ImGuiConfigFlags_NoMouse)
    .value("NO_MOUSE_CURSOR_CHANGE", ImGuiConfigFlags_NoMouseCursorChange)
    .value("NO_KEYBOARD", ImGuiConfigFlags_NoKeyboard)
    .value("DOCKING_ENABLE", ImGuiConfigFlags_DockingEnable)
    .value("VIEWPORTS_ENABLE", ImGuiConfigFlags_ViewportsEnable)
    .value("DPI_ENABLE_SCALE_VIEWPORTS", ImGuiConfigFlags_DpiEnableScaleViewports)
    .value("DPI_ENABLE_SCALE_FONTS", ImGuiConfigFlags_DpiEnableScaleFonts)
    .value("IS_S_R_G_B", ImGuiConfigFlags_IsSRGB)
    .value("IS_TOUCH_SCREEN", ImGuiConfigFlags_IsTouchScreen)
    .value("NAV_ENABLE_SET_MOUSE_POS", ImGuiConfigFlags_NavEnableSetMousePos)
    .value("NAV_NO_CAPTURE_KEYBOARD", ImGuiConfigFlags_NavNoCaptureKeyboard)
    ;

nb::enum_<ImGuiBackendFlags_>(mod_imgui, "BackendFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImGuiBackendFlags_None)
    .value("HAS_GAMEPAD", ImGuiBackendFlags_HasGamepad)
    .value("HAS_MOUSE_CURSORS", ImGuiBackendFlags_HasMouseCursors)
    .value("HAS_SET_MOUSE_POS", ImGuiBackendFlags_HasSetMousePos)
    .value("RENDERER_HAS_VTX_OFFSET", ImGuiBackendFlags_RendererHasVtxOffset)
    .value("PLATFORM_HAS_VIEWPORTS", ImGuiBackendFlags_PlatformHasViewports)
    .value("HAS_MOUSE_HOVERED_VIEWPORT", ImGuiBackendFlags_HasMouseHoveredViewport)
    .value("RENDERER_HAS_VIEWPORTS", ImGuiBackendFlags_RendererHasViewports)
    ;

nb::enum_<ImGuiCol_>(mod_imgui, "Col", nb::is_arithmetic() )
    .value("TEXT", ImGuiCol_Text)
    .value("TEXT_DISABLED", ImGuiCol_TextDisabled)
    .value("WINDOW_BG", ImGuiCol_WindowBg)
    .value("CHILD_BG", ImGuiCol_ChildBg)
    .value("POPUP_BG", ImGuiCol_PopupBg)
    .value("BORDER", ImGuiCol_Border)
    .value("BORDER_SHADOW", ImGuiCol_BorderShadow)
    .value("FRAME_BG", ImGuiCol_FrameBg)
    .value("FRAME_BG_HOVERED", ImGuiCol_FrameBgHovered)
    .value("FRAME_BG_ACTIVE", ImGuiCol_FrameBgActive)
    .value("TITLE_BG", ImGuiCol_TitleBg)
    .value("TITLE_BG_ACTIVE", ImGuiCol_TitleBgActive)
    .value("TITLE_BG_COLLAPSED", ImGuiCol_TitleBgCollapsed)
    .value("MENU_BAR_BG", ImGuiCol_MenuBarBg)
    .value("SCROLLBAR_BG", ImGuiCol_ScrollbarBg)
    .value("SCROLLBAR_GRAB", ImGuiCol_ScrollbarGrab)
    .value("SCROLLBAR_GRAB_HOVERED", ImGuiCol_ScrollbarGrabHovered)
    .value("SCROLLBAR_GRAB_ACTIVE", ImGuiCol_ScrollbarGrabActive)
    .value("CHECK_MARK", ImGuiCol_CheckMark)
    .value("SLIDER_GRAB", ImGuiCol_SliderGrab)
    .value("SLIDER_GRAB_ACTIVE", ImGuiCol_SliderGrabActive)
    .value("BUTTON", ImGuiCol_Button)
    .value("BUTTON_HOVERED", ImGuiCol_ButtonHovered)
    .value("BUTTON_ACTIVE", ImGuiCol_ButtonActive)
    .value("HEADER", ImGuiCol_Header)
    .value("HEADER_HOVERED", ImGuiCol_HeaderHovered)
    .value("HEADER_ACTIVE", ImGuiCol_HeaderActive)
    .value("SEPARATOR", ImGuiCol_Separator)
    .value("SEPARATOR_HOVERED", ImGuiCol_SeparatorHovered)
    .value("SEPARATOR_ACTIVE", ImGuiCol_SeparatorActive)
    .value("RESIZE_GRIP", ImGuiCol_ResizeGrip)
    .value("RESIZE_GRIP_HOVERED", ImGuiCol_ResizeGripHovered)
    .value("RESIZE_GRIP_ACTIVE", ImGuiCol_ResizeGripActive)
    .value("TAB_HOVERED", ImGuiCol_TabHovered)
    .value("TAB", ImGuiCol_Tab)
    .value("TAB_SELECTED", ImGuiCol_TabSelected)
    .value("TAB_SELECTED_OVERLINE", ImGuiCol_TabSelectedOverline)
    .value("TAB_DIMMED", ImGuiCol_TabDimmed)
    .value("TAB_DIMMED_SELECTED", ImGuiCol_TabDimmedSelected)
    .value("TAB_DIMMED_SELECTED_OVERLINE", ImGuiCol_TabDimmedSelectedOverline)
    .value("DOCKING_PREVIEW", ImGuiCol_DockingPreview)
    .value("DOCKING_EMPTY_BG", ImGuiCol_DockingEmptyBg)
    .value("PLOT_LINES", ImGuiCol_PlotLines)
    .value("PLOT_LINES_HOVERED", ImGuiCol_PlotLinesHovered)
    .value("PLOT_HISTOGRAM", ImGuiCol_PlotHistogram)
    .value("PLOT_HISTOGRAM_HOVERED", ImGuiCol_PlotHistogramHovered)
    .value("TABLE_HEADER_BG", ImGuiCol_TableHeaderBg)
    .value("TABLE_BORDER_STRONG", ImGuiCol_TableBorderStrong)
    .value("TABLE_BORDER_LIGHT", ImGuiCol_TableBorderLight)
    .value("TABLE_ROW_BG", ImGuiCol_TableRowBg)
    .value("TABLE_ROW_BG_ALT", ImGuiCol_TableRowBgAlt)
    .value("TEXT_LINK", ImGuiCol_TextLink)
    .value("TEXT_SELECTED_BG", ImGuiCol_TextSelectedBg)
    .value("DRAG_DROP_TARGET", ImGuiCol_DragDropTarget)
    .value("NAV_CURSOR", ImGuiCol_NavCursor)
    .value("NAV_WINDOWING_HIGHLIGHT", ImGuiCol_NavWindowingHighlight)
    .value("NAV_WINDOWING_DIM_BG", ImGuiCol_NavWindowingDimBg)
    .value("MODAL_WINDOW_DIM_BG", ImGuiCol_ModalWindowDimBg)
    .value("TAB_ACTIVE", ImGuiCol_TabActive)
    .value("TAB_UNFOCUSED", ImGuiCol_TabUnfocused)
    .value("TAB_UNFOCUSED_ACTIVE", ImGuiCol_TabUnfocusedActive)
    .value("NAV_HIGHLIGHT", ImGuiCol_NavHighlight)
    ;

nb::enum_<ImGuiStyleVar_>(mod_imgui, "StyleVar", nb::is_arithmetic() )
    .value("ALPHA", ImGuiStyleVar_Alpha)
    .value("DISABLED_ALPHA", ImGuiStyleVar_DisabledAlpha)
    .value("WINDOW_PADDING", ImGuiStyleVar_WindowPadding)
    .value("WINDOW_ROUNDING", ImGuiStyleVar_WindowRounding)
    .value("WINDOW_BORDER_SIZE", ImGuiStyleVar_WindowBorderSize)
    .value("WINDOW_MIN_SIZE", ImGuiStyleVar_WindowMinSize)
    .value("WINDOW_TITLE_ALIGN", ImGuiStyleVar_WindowTitleAlign)
    .value("CHILD_ROUNDING", ImGuiStyleVar_ChildRounding)
    .value("CHILD_BORDER_SIZE", ImGuiStyleVar_ChildBorderSize)
    .value("POPUP_ROUNDING", ImGuiStyleVar_PopupRounding)
    .value("POPUP_BORDER_SIZE", ImGuiStyleVar_PopupBorderSize)
    .value("FRAME_PADDING", ImGuiStyleVar_FramePadding)
    .value("FRAME_ROUNDING", ImGuiStyleVar_FrameRounding)
    .value("FRAME_BORDER_SIZE", ImGuiStyleVar_FrameBorderSize)
    .value("ITEM_SPACING", ImGuiStyleVar_ItemSpacing)
    .value("ITEM_INNER_SPACING", ImGuiStyleVar_ItemInnerSpacing)
    .value("INDENT_SPACING", ImGuiStyleVar_IndentSpacing)
    .value("CELL_PADDING", ImGuiStyleVar_CellPadding)
    .value("SCROLLBAR_SIZE", ImGuiStyleVar_ScrollbarSize)
    .value("SCROLLBAR_ROUNDING", ImGuiStyleVar_ScrollbarRounding)
    .value("GRAB_MIN_SIZE", ImGuiStyleVar_GrabMinSize)
    .value("GRAB_ROUNDING", ImGuiStyleVar_GrabRounding)
    .value("IMAGE_BORDER_SIZE", ImGuiStyleVar_ImageBorderSize)
    .value("TAB_ROUNDING", ImGuiStyleVar_TabRounding)
    .value("TAB_BORDER_SIZE", ImGuiStyleVar_TabBorderSize)
    .value("TAB_BAR_BORDER_SIZE", ImGuiStyleVar_TabBarBorderSize)
    .value("TAB_BAR_OVERLINE_SIZE", ImGuiStyleVar_TabBarOverlineSize)
    .value("TABLE_ANGLED_HEADERS_ANGLE", ImGuiStyleVar_TableAngledHeadersAngle)
    .value("TABLE_ANGLED_HEADERS_TEXT_ALIGN", ImGuiStyleVar_TableAngledHeadersTextAlign)
    .value("BUTTON_TEXT_ALIGN", ImGuiStyleVar_ButtonTextAlign)
    .value("SELECTABLE_TEXT_ALIGN", ImGuiStyleVar_SelectableTextAlign)
    .value("SEPARATOR_TEXT_BORDER_SIZE", ImGuiStyleVar_SeparatorTextBorderSize)
    .value("SEPARATOR_TEXT_ALIGN", ImGuiStyleVar_SeparatorTextAlign)
    .value("SEPARATOR_TEXT_PADDING", ImGuiStyleVar_SeparatorTextPadding)
    .value("DOCKING_SEPARATOR_SIZE", ImGuiStyleVar_DockingSeparatorSize)
    ;

nb::enum_<ImGuiButtonFlags_>(mod_imgui, "ButtonFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImGuiButtonFlags_None)
    .value("MOUSE_BUTTON_LEFT", ImGuiButtonFlags_MouseButtonLeft)
    .value("MOUSE_BUTTON_RIGHT", ImGuiButtonFlags_MouseButtonRight)
    .value("MOUSE_BUTTON_MIDDLE", ImGuiButtonFlags_MouseButtonMiddle)
    .value("MOUSE_BUTTON_MASK_", ImGuiButtonFlags_MouseButtonMask_)
    .value("ENABLE_NAV", ImGuiButtonFlags_EnableNav)
    ;

nb::enum_<ImGuiColorEditFlags_>(mod_imgui, "ColorEditFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImGuiColorEditFlags_None)
    .value("NO_ALPHA", ImGuiColorEditFlags_NoAlpha)
    .value("NO_PICKER", ImGuiColorEditFlags_NoPicker)
    .value("NO_OPTIONS", ImGuiColorEditFlags_NoOptions)
    .value("NO_SMALL_PREVIEW", ImGuiColorEditFlags_NoSmallPreview)
    .value("NO_INPUTS", ImGuiColorEditFlags_NoInputs)
    .value("NO_TOOLTIP", ImGuiColorEditFlags_NoTooltip)
    .value("NO_LABEL", ImGuiColorEditFlags_NoLabel)
    .value("NO_SIDE_PREVIEW", ImGuiColorEditFlags_NoSidePreview)
    .value("NO_DRAG_DROP", ImGuiColorEditFlags_NoDragDrop)
    .value("NO_BORDER", ImGuiColorEditFlags_NoBorder)
    .value("ALPHA_OPAQUE", ImGuiColorEditFlags_AlphaOpaque)
    .value("ALPHA_NO_BG", ImGuiColorEditFlags_AlphaNoBg)
    .value("ALPHA_PREVIEW_HALF", ImGuiColorEditFlags_AlphaPreviewHalf)
    .value("ALPHA_BAR", ImGuiColorEditFlags_AlphaBar)
    .value("H_D_R", ImGuiColorEditFlags_HDR)
    .value("DISPLAY_R_G_B", ImGuiColorEditFlags_DisplayRGB)
    .value("DISPLAY_H_S_V", ImGuiColorEditFlags_DisplayHSV)
    .value("DISPLAY_HEX", ImGuiColorEditFlags_DisplayHex)
    .value("UINT8", ImGuiColorEditFlags_Uint8)
    .value("FLOAT", ImGuiColorEditFlags_Float)
    .value("PICKER_HUE_BAR", ImGuiColorEditFlags_PickerHueBar)
    .value("PICKER_HUE_WHEEL", ImGuiColorEditFlags_PickerHueWheel)
    .value("INPUT_R_G_B", ImGuiColorEditFlags_InputRGB)
    .value("INPUT_H_S_V", ImGuiColorEditFlags_InputHSV)
    .value("DEFAULT_OPTIONS_", ImGuiColorEditFlags_DefaultOptions_)
    .value("ALPHA_MASK_", ImGuiColorEditFlags_AlphaMask_)
    .value("DISPLAY_MASK_", ImGuiColorEditFlags_DisplayMask_)
    .value("DATA_TYPE_MASK_", ImGuiColorEditFlags_DataTypeMask_)
    .value("PICKER_MASK_", ImGuiColorEditFlags_PickerMask_)
    .value("INPUT_MASK_", ImGuiColorEditFlags_InputMask_)
    .value("ALPHA_PREVIEW", ImGuiColorEditFlags_AlphaPreview)
    ;

nb::enum_<ImGuiSliderFlags_>(mod_imgui, "SliderFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImGuiSliderFlags_None)
    .value("LOGARITHMIC", ImGuiSliderFlags_Logarithmic)
    .value("NO_ROUND_TO_FORMAT", ImGuiSliderFlags_NoRoundToFormat)
    .value("NO_INPUT", ImGuiSliderFlags_NoInput)
    .value("WRAP_AROUND", ImGuiSliderFlags_WrapAround)
    .value("CLAMP_ON_INPUT", ImGuiSliderFlags_ClampOnInput)
    .value("CLAMP_ZERO_RANGE", ImGuiSliderFlags_ClampZeroRange)
    .value("NO_SPEED_TWEAKS", ImGuiSliderFlags_NoSpeedTweaks)
    .value("ALWAYS_CLAMP", ImGuiSliderFlags_AlwaysClamp)
    .value("INVALID_MASK_", ImGuiSliderFlags_InvalidMask_)
    ;

nb::enum_<ImGuiMouseButton_>(mod_imgui, "MouseButton", nb::is_arithmetic() )
    .value("LEFT", ImGuiMouseButton_Left)
    .value("RIGHT", ImGuiMouseButton_Right)
    .value("MIDDLE", ImGuiMouseButton_Middle)
    ;

nb::enum_<ImGuiMouseCursor_>(mod_imgui, "MouseCursor", nb::is_arithmetic() )
    .value("NONE", ImGuiMouseCursor_None)
    .value("ARROW", ImGuiMouseCursor_Arrow)
    .value("TEXT_INPUT", ImGuiMouseCursor_TextInput)
    .value("RESIZE_ALL", ImGuiMouseCursor_ResizeAll)
    .value("RESIZE_N_S", ImGuiMouseCursor_ResizeNS)
    .value("RESIZE_E_W", ImGuiMouseCursor_ResizeEW)
    .value("RESIZE_N_E_S_W", ImGuiMouseCursor_ResizeNESW)
    .value("RESIZE_N_W_S_E", ImGuiMouseCursor_ResizeNWSE)
    .value("HAND", ImGuiMouseCursor_Hand)
    .value("WAIT", ImGuiMouseCursor_Wait)
    .value("PROGRESS", ImGuiMouseCursor_Progress)
    .value("NOT_ALLOWED", ImGuiMouseCursor_NotAllowed)
    ;

nb::enum_<ImGuiMouseSource>(mod_imgui, "MouseSource", nb::is_arithmetic() )
    .value("MOUSE", ImGuiMouseSource_Mouse)
    .value("TOUCH_SCREEN", ImGuiMouseSource_TouchScreen)
    .value("PEN", ImGuiMouseSource_Pen)
    ;

nb::enum_<ImGuiCond_>(mod_imgui, "Cond", nb::is_arithmetic() )
    .value("NONE", ImGuiCond_None)
    .value("ALWAYS", ImGuiCond_Always)
    .value("ONCE", ImGuiCond_Once)
    .value("FIRST_USE_EVER", ImGuiCond_FirstUseEver)
    .value("APPEARING", ImGuiCond_Appearing)
    ;

nb::enum_<ImGuiTableFlags_>(mod_imgui, "TableFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImGuiTableFlags_None)
    .value("RESIZABLE", ImGuiTableFlags_Resizable)
    .value("REORDERABLE", ImGuiTableFlags_Reorderable)
    .value("HIDEABLE", ImGuiTableFlags_Hideable)
    .value("SORTABLE", ImGuiTableFlags_Sortable)
    .value("NO_SAVED_SETTINGS", ImGuiTableFlags_NoSavedSettings)
    .value("CONTEXT_MENU_IN_BODY", ImGuiTableFlags_ContextMenuInBody)
    .value("ROW_BG", ImGuiTableFlags_RowBg)
    .value("BORDERS_INNER_H", ImGuiTableFlags_BordersInnerH)
    .value("BORDERS_OUTER_H", ImGuiTableFlags_BordersOuterH)
    .value("BORDERS_INNER_V", ImGuiTableFlags_BordersInnerV)
    .value("BORDERS_OUTER_V", ImGuiTableFlags_BordersOuterV)
    .value("BORDERS_H", ImGuiTableFlags_BordersH)
    .value("BORDERS_V", ImGuiTableFlags_BordersV)
    .value("BORDERS_INNER", ImGuiTableFlags_BordersInner)
    .value("BORDERS_OUTER", ImGuiTableFlags_BordersOuter)
    .value("BORDERS", ImGuiTableFlags_Borders)
    .value("NO_BORDERS_IN_BODY", ImGuiTableFlags_NoBordersInBody)
    .value("NO_BORDERS_IN_BODY_UNTIL_RESIZE", ImGuiTableFlags_NoBordersInBodyUntilResize)
    .value("SIZING_FIXED_FIT", ImGuiTableFlags_SizingFixedFit)
    .value("SIZING_FIXED_SAME", ImGuiTableFlags_SizingFixedSame)
    .value("SIZING_STRETCH_PROP", ImGuiTableFlags_SizingStretchProp)
    .value("SIZING_STRETCH_SAME", ImGuiTableFlags_SizingStretchSame)
    .value("NO_HOST_EXTEND_X", ImGuiTableFlags_NoHostExtendX)
    .value("NO_HOST_EXTEND_Y", ImGuiTableFlags_NoHostExtendY)
    .value("NO_KEEP_COLUMNS_VISIBLE", ImGuiTableFlags_NoKeepColumnsVisible)
    .value("PRECISE_WIDTHS", ImGuiTableFlags_PreciseWidths)
    .value("NO_CLIP", ImGuiTableFlags_NoClip)
    .value("PAD_OUTER_X", ImGuiTableFlags_PadOuterX)
    .value("NO_PAD_OUTER_X", ImGuiTableFlags_NoPadOuterX)
    .value("NO_PAD_INNER_X", ImGuiTableFlags_NoPadInnerX)
    .value("SCROLL_X", ImGuiTableFlags_ScrollX)
    .value("SCROLL_Y", ImGuiTableFlags_ScrollY)
    .value("SORT_MULTI", ImGuiTableFlags_SortMulti)
    .value("SORT_TRISTATE", ImGuiTableFlags_SortTristate)
    .value("HIGHLIGHT_HOVERED_COLUMN", ImGuiTableFlags_HighlightHoveredColumn)
    .value("SIZING_MASK_", ImGuiTableFlags_SizingMask_)
    ;

nb::enum_<ImGuiTableColumnFlags_>(mod_imgui, "TableColumnFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImGuiTableColumnFlags_None)
    .value("DISABLED", ImGuiTableColumnFlags_Disabled)
    .value("DEFAULT_HIDE", ImGuiTableColumnFlags_DefaultHide)
    .value("DEFAULT_SORT", ImGuiTableColumnFlags_DefaultSort)
    .value("WIDTH_STRETCH", ImGuiTableColumnFlags_WidthStretch)
    .value("WIDTH_FIXED", ImGuiTableColumnFlags_WidthFixed)
    .value("NO_RESIZE", ImGuiTableColumnFlags_NoResize)
    .value("NO_REORDER", ImGuiTableColumnFlags_NoReorder)
    .value("NO_HIDE", ImGuiTableColumnFlags_NoHide)
    .value("NO_CLIP", ImGuiTableColumnFlags_NoClip)
    .value("NO_SORT", ImGuiTableColumnFlags_NoSort)
    .value("NO_SORT_ASCENDING", ImGuiTableColumnFlags_NoSortAscending)
    .value("NO_SORT_DESCENDING", ImGuiTableColumnFlags_NoSortDescending)
    .value("NO_HEADER_LABEL", ImGuiTableColumnFlags_NoHeaderLabel)
    .value("NO_HEADER_WIDTH", ImGuiTableColumnFlags_NoHeaderWidth)
    .value("PREFER_SORT_ASCENDING", ImGuiTableColumnFlags_PreferSortAscending)
    .value("PREFER_SORT_DESCENDING", ImGuiTableColumnFlags_PreferSortDescending)
    .value("INDENT_ENABLE", ImGuiTableColumnFlags_IndentEnable)
    .value("INDENT_DISABLE", ImGuiTableColumnFlags_IndentDisable)
    .value("ANGLED_HEADER", ImGuiTableColumnFlags_AngledHeader)
    .value("IS_ENABLED", ImGuiTableColumnFlags_IsEnabled)
    .value("IS_VISIBLE", ImGuiTableColumnFlags_IsVisible)
    .value("IS_SORTED", ImGuiTableColumnFlags_IsSorted)
    .value("IS_HOVERED", ImGuiTableColumnFlags_IsHovered)
    .value("WIDTH_MASK_", ImGuiTableColumnFlags_WidthMask_)
    .value("INDENT_MASK_", ImGuiTableColumnFlags_IndentMask_)
    .value("STATUS_MASK_", ImGuiTableColumnFlags_StatusMask_)
    .value("NO_DIRECT_RESIZE_", ImGuiTableColumnFlags_NoDirectResize_)
    ;

nb::enum_<ImGuiTableRowFlags_>(mod_imgui, "TableRowFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImGuiTableRowFlags_None)
    .value("HEADERS", ImGuiTableRowFlags_Headers)
    ;

nb::enum_<ImGuiTableBgTarget_>(mod_imgui, "TableBgTarget", nb::is_arithmetic() )
    .value("NONE", ImGuiTableBgTarget_None)
    .value("ROW_BG0", ImGuiTableBgTarget_RowBg0)
    .value("ROW_BG1", ImGuiTableBgTarget_RowBg1)
    .value("CELL_BG", ImGuiTableBgTarget_CellBg)
    ;

nb::enum_<ImGuiMultiSelectFlags_>(mod_imgui, "MultiSelectFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImGuiMultiSelectFlags_None)
    .value("SINGLE_SELECT", ImGuiMultiSelectFlags_SingleSelect)
    .value("NO_SELECT_ALL", ImGuiMultiSelectFlags_NoSelectAll)
    .value("NO_RANGE_SELECT", ImGuiMultiSelectFlags_NoRangeSelect)
    .value("NO_AUTO_SELECT", ImGuiMultiSelectFlags_NoAutoSelect)
    .value("NO_AUTO_CLEAR", ImGuiMultiSelectFlags_NoAutoClear)
    .value("NO_AUTO_CLEAR_ON_RESELECT", ImGuiMultiSelectFlags_NoAutoClearOnReselect)
    .value("BOX_SELECT1D", ImGuiMultiSelectFlags_BoxSelect1d)
    .value("BOX_SELECT2D", ImGuiMultiSelectFlags_BoxSelect2d)
    .value("BOX_SELECT_NO_SCROLL", ImGuiMultiSelectFlags_BoxSelectNoScroll)
    .value("CLEAR_ON_ESCAPE", ImGuiMultiSelectFlags_ClearOnEscape)
    .value("CLEAR_ON_CLICK_VOID", ImGuiMultiSelectFlags_ClearOnClickVoid)
    .value("SCOPE_WINDOW", ImGuiMultiSelectFlags_ScopeWindow)
    .value("SCOPE_RECT", ImGuiMultiSelectFlags_ScopeRect)
    .value("SELECT_ON_CLICK", ImGuiMultiSelectFlags_SelectOnClick)
    .value("SELECT_ON_CLICK_RELEASE", ImGuiMultiSelectFlags_SelectOnClickRelease)
    .value("NAV_WRAP_X", ImGuiMultiSelectFlags_NavWrapX)
    ;

nb::enum_<ImGuiSelectionRequestType>(mod_imgui, "SelectionRequestType", nb::is_arithmetic() )
    .value("NONE", ImGuiSelectionRequestType_None)
    .value("SET_ALL", ImGuiSelectionRequestType_SetAll)
    .value("SET_RANGE", ImGuiSelectionRequestType_SetRange)
    ;

nb::enum_<ImDrawFlags_>(mod_imgui, "DrawFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImDrawFlags_None)
    .value("CLOSED", ImDrawFlags_Closed)
    .value("ROUND_CORNERS_TOP_LEFT", ImDrawFlags_RoundCornersTopLeft)
    .value("ROUND_CORNERS_TOP_RIGHT", ImDrawFlags_RoundCornersTopRight)
    .value("ROUND_CORNERS_BOTTOM_LEFT", ImDrawFlags_RoundCornersBottomLeft)
    .value("ROUND_CORNERS_BOTTOM_RIGHT", ImDrawFlags_RoundCornersBottomRight)
    .value("ROUND_CORNERS_NONE", ImDrawFlags_RoundCornersNone)
    .value("ROUND_CORNERS_TOP", ImDrawFlags_RoundCornersTop)
    .value("ROUND_CORNERS_BOTTOM", ImDrawFlags_RoundCornersBottom)
    .value("ROUND_CORNERS_LEFT", ImDrawFlags_RoundCornersLeft)
    .value("ROUND_CORNERS_RIGHT", ImDrawFlags_RoundCornersRight)
    .value("ROUND_CORNERS_ALL", ImDrawFlags_RoundCornersAll)
    .value("ROUND_CORNERS_DEFAULT_", ImDrawFlags_RoundCornersDefault_)
    .value("ROUND_CORNERS_MASK_", ImDrawFlags_RoundCornersMask_)
    ;

nb::enum_<ImDrawListFlags_>(mod_imgui, "DrawListFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImDrawListFlags_None)
    .value("ANTI_ALIASED_LINES", ImDrawListFlags_AntiAliasedLines)
    .value("ANTI_ALIASED_LINES_USE_TEX", ImDrawListFlags_AntiAliasedLinesUseTex)
    .value("ANTI_ALIASED_FILL", ImDrawListFlags_AntiAliasedFill)
    .value("ALLOW_VTX_OFFSET", ImDrawListFlags_AllowVtxOffset)
    ;

nb::enum_<ImFontAtlasFlags_>(mod_imgui, "FontAtlasFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImFontAtlasFlags_None)
    .value("NO_POWER_OF_TWO_HEIGHT", ImFontAtlasFlags_NoPowerOfTwoHeight)
    .value("NO_MOUSE_CURSORS", ImFontAtlasFlags_NoMouseCursors)
    .value("NO_BAKED_LINES", ImFontAtlasFlags_NoBakedLines)
    ;

nb::enum_<ImGuiViewportFlags_>(mod_imgui, "ViewportFlags", nb::is_arithmetic() , nb::is_flag())
    .value("NONE", ImGuiViewportFlags_None)
    .value("IS_PLATFORM_WINDOW", ImGuiViewportFlags_IsPlatformWindow)
    .value("IS_PLATFORM_MONITOR", ImGuiViewportFlags_IsPlatformMonitor)
    .value("OWNED_BY_APP", ImGuiViewportFlags_OwnedByApp)
    .value("NO_DECORATION", ImGuiViewportFlags_NoDecoration)
    .value("NO_TASK_BAR_ICON", ImGuiViewportFlags_NoTaskBarIcon)
    .value("NO_FOCUS_ON_APPEARING", ImGuiViewportFlags_NoFocusOnAppearing)
    .value("NO_FOCUS_ON_CLICK", ImGuiViewportFlags_NoFocusOnClick)
    .value("NO_INPUTS", ImGuiViewportFlags_NoInputs)
    .value("NO_RENDERER_CLEAR", ImGuiViewportFlags_NoRendererClear)
    .value("NO_AUTO_MERGE", ImGuiViewportFlags_NoAutoMerge)
    .value("TOP_MOST", ImGuiViewportFlags_TopMost)
    .value("CAN_HOST_OTHER_WINDOWS", ImGuiViewportFlags_CanHostOtherWindows)
    .value("IS_MINIMIZED", ImGuiViewportFlags_IsMinimized)
    .value("IS_FOCUSED", ImGuiViewportFlags_IsFocused)
    ;

nb::class_<ImGuiIO>(mod_imgui, "IO")
    .def_rw("delta_time", &ImGuiIO::DeltaTime)
    .def_rw("ini_saving_rate", &ImGuiIO::IniSavingRate)
    .def_rw("font_global_scale", &ImGuiIO::FontGlobalScale)
    .def_rw("font_allow_user_scaling", &ImGuiIO::FontAllowUserScaling)
    .def_rw("config_nav_swap_gamepad_buttons", &ImGuiIO::ConfigNavSwapGamepadButtons)
    .def_rw("config_nav_move_set_mouse_pos", &ImGuiIO::ConfigNavMoveSetMousePos)
    .def_rw("config_nav_capture_keyboard", &ImGuiIO::ConfigNavCaptureKeyboard)
    .def_rw("config_nav_escape_clear_focus_item", &ImGuiIO::ConfigNavEscapeClearFocusItem)
    .def_rw("config_nav_escape_clear_focus_window", &ImGuiIO::ConfigNavEscapeClearFocusWindow)
    .def_rw("config_nav_cursor_visible_auto", &ImGuiIO::ConfigNavCursorVisibleAuto)
    .def_rw("config_nav_cursor_visible_always", &ImGuiIO::ConfigNavCursorVisibleAlways)
    .def_rw("config_docking_no_split", &ImGuiIO::ConfigDockingNoSplit)
    .def_rw("config_docking_with_shift", &ImGuiIO::ConfigDockingWithShift)
    .def_rw("config_docking_always_tab_bar", &ImGuiIO::ConfigDockingAlwaysTabBar)
    .def_rw("config_docking_transparent_payload", &ImGuiIO::ConfigDockingTransparentPayload)
    .def_rw("config_viewports_no_auto_merge", &ImGuiIO::ConfigViewportsNoAutoMerge)
    .def_rw("config_viewports_no_task_bar_icon", &ImGuiIO::ConfigViewportsNoTaskBarIcon)
    .def_rw("config_viewports_no_decoration", &ImGuiIO::ConfigViewportsNoDecoration)
    .def_rw("config_viewports_no_default_parent", &ImGuiIO::ConfigViewportsNoDefaultParent)
    .def_rw("mouse_draw_cursor", &ImGuiIO::MouseDrawCursor)
    .def_rw("config_mac_o_s_x_behaviors", &ImGuiIO::ConfigMacOSXBehaviors)
    .def_rw("config_input_trickle_event_queue", &ImGuiIO::ConfigInputTrickleEventQueue)
    .def_rw("config_input_text_cursor_blink", &ImGuiIO::ConfigInputTextCursorBlink)
    .def_rw("config_input_text_enter_keep_active", &ImGuiIO::ConfigInputTextEnterKeepActive)
    .def_rw("config_drag_click_to_input_text", &ImGuiIO::ConfigDragClickToInputText)
    .def_rw("config_windows_resize_from_edges", &ImGuiIO::ConfigWindowsResizeFromEdges)
    .def_rw("config_windows_move_from_title_bar_only", &ImGuiIO::ConfigWindowsMoveFromTitleBarOnly)
    .def_rw("config_windows_copy_contents_with_ctrl_c", &ImGuiIO::ConfigWindowsCopyContentsWithCtrlC)
    .def_rw("config_scrollbar_scroll_by_page", &ImGuiIO::ConfigScrollbarScrollByPage)
    .def_rw("config_memory_compact_timer", &ImGuiIO::ConfigMemoryCompactTimer)
    .def_rw("mouse_double_click_time", &ImGuiIO::MouseDoubleClickTime)
    .def_rw("mouse_double_click_max_dist", &ImGuiIO::MouseDoubleClickMaxDist)
    .def_rw("mouse_drag_threshold", &ImGuiIO::MouseDragThreshold)
    .def_rw("key_repeat_delay", &ImGuiIO::KeyRepeatDelay)
    .def_rw("key_repeat_rate", &ImGuiIO::KeyRepeatRate)
    .def_rw("config_error_recovery", &ImGuiIO::ConfigErrorRecovery)
    .def_rw("config_error_recovery_enable_assert", &ImGuiIO::ConfigErrorRecoveryEnableAssert)
    .def_rw("config_error_recovery_enable_debug_log", &ImGuiIO::ConfigErrorRecoveryEnableDebugLog)
    .def_rw("config_error_recovery_enable_tooltip", &ImGuiIO::ConfigErrorRecoveryEnableTooltip)
    .def_rw("config_debug_is_debugger_present", &ImGuiIO::ConfigDebugIsDebuggerPresent)
    .def_rw("config_debug_highlight_id_conflicts", &ImGuiIO::ConfigDebugHighlightIdConflicts)
    .def_rw("config_debug_highlight_id_conflicts_show_item_picker", &ImGuiIO::ConfigDebugHighlightIdConflictsShowItemPicker)
    .def_rw("config_debug_begin_return_value_once", &ImGuiIO::ConfigDebugBeginReturnValueOnce)
    .def_rw("config_debug_begin_return_value_loop", &ImGuiIO::ConfigDebugBeginReturnValueLoop)
    .def_rw("config_debug_ignore_focus_loss", &ImGuiIO::ConfigDebugIgnoreFocusLoss)
    .def_rw("config_debug_ini_settings", &ImGuiIO::ConfigDebugIniSettings)
    .def_rw("want_capture_mouse", &ImGuiIO::WantCaptureMouse)
    .def_rw("want_capture_keyboard", &ImGuiIO::WantCaptureKeyboard)
    .def_rw("want_text_input", &ImGuiIO::WantTextInput)
    .def_rw("want_set_mouse_pos", &ImGuiIO::WantSetMousePos)
    .def_rw("want_save_ini_settings", &ImGuiIO::WantSaveIniSettings)
    .def_rw("nav_active", &ImGuiIO::NavActive)
    .def_rw("nav_visible", &ImGuiIO::NavVisible)
    .def_rw("framerate", &ImGuiIO::Framerate)
    .def_rw("metrics_render_vertices", &ImGuiIO::MetricsRenderVertices)
    .def_rw("metrics_render_indices", &ImGuiIO::MetricsRenderIndices)
    .def_rw("metrics_render_windows", &ImGuiIO::MetricsRenderWindows)
    .def_rw("metrics_active_windows", &ImGuiIO::MetricsActiveWindows)
    .def_rw("mouse_wheel", &ImGuiIO::MouseWheel)
    .def_rw("mouse_wheel_h", &ImGuiIO::MouseWheelH)
    .def_rw("key_ctrl", &ImGuiIO::KeyCtrl)
    .def_rw("key_shift", &ImGuiIO::KeyShift)
    .def_rw("key_alt", &ImGuiIO::KeyAlt)
    .def_rw("key_super", &ImGuiIO::KeySuper)
    .def_rw("want_capture_mouse_unless_popup_close", &ImGuiIO::WantCaptureMouseUnlessPopupClose)
    .def_rw("mouse_wheel_request_axis_swap", &ImGuiIO::MouseWheelRequestAxisSwap)
    .def_rw("mouse_ctrl_left_as_right_click", &ImGuiIO::MouseCtrlLeftAsRightClick)
    .def_rw("pen_pressure", &ImGuiIO::PenPressure)
    .def_rw("app_focus_lost", &ImGuiIO::AppFocusLost)
    .def_rw("app_accepting_events", &ImGuiIO::AppAcceptingEvents)
;

mod_imgui.def("get_style",
    [] () -> ImGuiStyle& {
        return ImGui::GetStyle();
    });

mod_imgui.def("show_demo_window",
    [] (std::optional<bool> p_open) -> std::optional<bool> {
        ImGui::ShowDemoWindow(p_open.has_value() ? &p_open.value() : NULL);
        return p_open;
    }, nb::arg("p_open") = nb::none());

mod_imgui.def("show_metrics_window",
    [] (std::optional<bool> p_open) -> std::optional<bool> {
        ImGui::ShowMetricsWindow(p_open.has_value() ? &p_open.value() : NULL);
        return p_open;
    }, nb::arg("p_open") = nb::none());

mod_imgui.def("show_debug_log_window",
    [] (std::optional<bool> p_open) -> std::optional<bool> {
        ImGui::ShowDebugLogWindow(p_open.has_value() ? &p_open.value() : NULL);
        return p_open;
    }, nb::arg("p_open") = nb::none());

mod_imgui.def("show_id_stack_tool_window",
    [] (std::optional<bool> p_open) -> std::optional<bool> {
        ImGui::ShowIDStackToolWindow(p_open.has_value() ? &p_open.value() : NULL);
        return p_open;
    }, nb::arg("p_open") = nb::none());

mod_imgui.def("show_about_window",
    [] (std::optional<bool> p_open) -> std::optional<bool> {
        ImGui::ShowAboutWindow(p_open.has_value() ? &p_open.value() : NULL);
        return p_open;
    }, nb::arg("p_open") = nb::none());

mod_imgui.def("show_style_editor",
    [] (std::optional<ImGuiStyle*> ref) -> void {
        return ImGui::ShowStyleEditor(ref.has_value() ? ref.value() : NULL);
    }, nb::arg("ref") = nb::none());

mod_imgui.def("show_style_selector",
    [] (const char* label) -> bool {
        return ImGui::ShowStyleSelector(label);
    }, nb::arg("label"));

mod_imgui.def("show_font_selector",
    [] (const char* label) -> void {
        return ImGui::ShowFontSelector(label);
    }, nb::arg("label"));

mod_imgui.def("show_user_guide",
    [] () -> void {
        return ImGui::ShowUserGuide();
    });

mod_imgui.def("get_version",
    [] () -> const char* {
        return ImGui::GetVersion();
    });

mod_imgui.def("style_colors_dark",
    [] (std::optional<ImGuiStyle*> dst) -> void {
        return ImGui::StyleColorsDark(dst.has_value() ? dst.value() : NULL);
    }, nb::arg("dst") = nb::none());

mod_imgui.def("style_colors_light",
    [] (std::optional<ImGuiStyle*> dst) -> void {
        return ImGui::StyleColorsLight(dst.has_value() ? dst.value() : NULL);
    }, nb::arg("dst") = nb::none());

mod_imgui.def("style_colors_classic",
    [] (std::optional<ImGuiStyle*> dst) -> void {
        return ImGui::StyleColorsClassic(dst.has_value() ? dst.value() : NULL);
    }, nb::arg("dst") = nb::none());

mod_imgui.def("begin",
    [] (const char* name, std::optional<bool> p_open, ImGuiWindowFlags flags) -> nb::tuple {
        auto _call = ImGui::Begin(name, p_open.has_value() ? &p_open.value() : NULL, flags);
        auto _ret = p_open;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("name"), nb::arg("p_open") = nb::none(), nb::arg("flags") = 0);

mod_imgui.def("end",
    [] () -> void {
        return ImGui::End();
    });

mod_imgui.def("begin_child",
    [] (const char* str_id, ImVec2 size, ImGuiChildFlags child_flags, ImGuiWindowFlags window_flags) -> bool {
        return ImGui::BeginChild(str_id, size, child_flags, window_flags);
    }, nb::arg("str_id"), nb::arg("size") = nb::make_tuple(0, 0), nb::arg("child_flags") = 0, nb::arg("window_flags") = 0);

mod_imgui.def("end_child",
    [] () -> void {
        return ImGui::EndChild();
    });

mod_imgui.def("is_window_appearing",
    [] () -> bool {
        return ImGui::IsWindowAppearing();
    });

mod_imgui.def("is_window_collapsed",
    [] () -> bool {
        return ImGui::IsWindowCollapsed();
    });

mod_imgui.def("is_window_focused",
    [] (ImGuiFocusedFlags flags) -> bool {
        return ImGui::IsWindowFocused(flags);
    }, nb::arg("flags") = 0);

mod_imgui.def("is_window_hovered",
    [] (ImGuiHoveredFlags flags) -> bool {
        return ImGui::IsWindowHovered(flags);
    }, nb::arg("flags") = 0);

mod_imgui.def("get_window_draw_list",
    [] () -> DrawList* {
        return new DrawList(ImGui::GetWindowDrawList());
    });

mod_imgui.def("get_window_dpi_scale",
    [] () -> float {
        return ImGui::GetWindowDpiScale();
    });

mod_imgui.def("get_window_pos",
    [] () -> ImVec2 {
        return ImGui::GetWindowPos();
    });

mod_imgui.def("get_window_size",
    [] () -> ImVec2 {
        return ImGui::GetWindowSize();
    });

mod_imgui.def("get_window_width",
    [] () -> float {
        return ImGui::GetWindowWidth();
    });

mod_imgui.def("get_window_height",
    [] () -> float {
        return ImGui::GetWindowHeight();
    });

mod_imgui.def("get_window_viewport",
    [] () -> ImGuiViewport* {
        return ImGui::GetWindowViewport();
    });

mod_imgui.def("set_next_window_pos",
    [] (ImVec2 pos, ImGuiCond cond, ImVec2 pivot) -> void {
        return ImGui::SetNextWindowPos(pos, cond, pivot);
    }, nb::arg("pos"), nb::arg("cond") = 0, nb::arg("pivot") = nb::make_tuple(0, 0));

mod_imgui.def("set_next_window_size",
    [] (ImVec2 size, ImGuiCond cond) -> void {
        return ImGui::SetNextWindowSize(size, cond);
    }, nb::arg("size"), nb::arg("cond") = 0);

mod_imgui.def("set_next_window_content_size",
    [] (ImVec2 size) -> void {
        return ImGui::SetNextWindowContentSize(size);
    }, nb::arg("size"));

mod_imgui.def("set_next_window_collapsed",
    [] (bool collapsed, ImGuiCond cond) -> void {
        return ImGui::SetNextWindowCollapsed(collapsed, cond);
    }, nb::arg("collapsed"), nb::arg("cond") = 0);

mod_imgui.def("set_next_window_focus",
    [] () -> void {
        return ImGui::SetNextWindowFocus();
    });

mod_imgui.def("set_next_window_scroll",
    [] (ImVec2 scroll) -> void {
        return ImGui::SetNextWindowScroll(scroll);
    }, nb::arg("scroll"));

mod_imgui.def("set_next_window_bg_alpha",
    [] (float alpha) -> void {
        return ImGui::SetNextWindowBgAlpha(alpha);
    }, nb::arg("alpha"));

mod_imgui.def("set_window_pos",
    [] (ImVec2 pos, ImGuiCond cond) -> void {
        return ImGui::SetWindowPos(pos, cond);
    }, nb::arg("pos"), nb::arg("cond") = 0);

mod_imgui.def("set_window_size",
    [] (ImVec2 size, ImGuiCond cond) -> void {
        return ImGui::SetWindowSize(size, cond);
    }, nb::arg("size"), nb::arg("cond") = 0);

mod_imgui.def("set_window_collapsed",
    [] (bool collapsed, ImGuiCond cond) -> void {
        return ImGui::SetWindowCollapsed(collapsed, cond);
    }, nb::arg("collapsed"), nb::arg("cond") = 0);

mod_imgui.def("set_window_focus",
    [] () -> void {
        return ImGui::SetWindowFocus();
    });

mod_imgui.def("set_window_font_scale",
    [] (float scale) -> void {
        return ImGui::SetWindowFontScale(scale);
    }, nb::arg("scale"));

mod_imgui.def("set_window_pos_str",
    [] (const char* name, ImVec2 pos, ImGuiCond cond) -> void {
        return ImGui::SetWindowPos(name, pos, cond);
    }, nb::arg("name"), nb::arg("pos"), nb::arg("cond") = 0);

mod_imgui.def("set_window_size_str",
    [] (const char* name, ImVec2 size, ImGuiCond cond) -> void {
        return ImGui::SetWindowSize(name, size, cond);
    }, nb::arg("name"), nb::arg("size"), nb::arg("cond") = 0);

mod_imgui.def("set_window_collapsed_str",
    [] (const char* name, bool collapsed, ImGuiCond cond) -> void {
        return ImGui::SetWindowCollapsed(name, collapsed, cond);
    }, nb::arg("name"), nb::arg("collapsed"), nb::arg("cond") = 0);

mod_imgui.def("set_window_focus_str",
    [] (const char* name) -> void {
        return ImGui::SetWindowFocus(name);
    }, nb::arg("name"));

mod_imgui.def("get_scroll_x",
    [] () -> float {
        return ImGui::GetScrollX();
    });

mod_imgui.def("get_scroll_y",
    [] () -> float {
        return ImGui::GetScrollY();
    });

mod_imgui.def("set_scroll_x",
    [] (float scroll_x) -> void {
        return ImGui::SetScrollX(scroll_x);
    }, nb::arg("scroll_x"));

mod_imgui.def("set_scroll_y",
    [] (float scroll_y) -> void {
        return ImGui::SetScrollY(scroll_y);
    }, nb::arg("scroll_y"));

mod_imgui.def("get_scroll_max_x",
    [] () -> float {
        return ImGui::GetScrollMaxX();
    });

mod_imgui.def("get_scroll_max_y",
    [] () -> float {
        return ImGui::GetScrollMaxY();
    });

mod_imgui.def("set_scroll_here_x",
    [] (float center_x_ratio) -> void {
        return ImGui::SetScrollHereX(center_x_ratio);
    }, nb::arg("center_x_ratio") = 0.5);

mod_imgui.def("set_scroll_here_y",
    [] (float center_y_ratio) -> void {
        return ImGui::SetScrollHereY(center_y_ratio);
    }, nb::arg("center_y_ratio") = 0.5);

mod_imgui.def("set_scroll_from_pos_x",
    [] (float local_x, float center_x_ratio) -> void {
        return ImGui::SetScrollFromPosX(local_x, center_x_ratio);
    }, nb::arg("local_x"), nb::arg("center_x_ratio") = 0.5);

mod_imgui.def("set_scroll_from_pos_y",
    [] (float local_y, float center_y_ratio) -> void {
        return ImGui::SetScrollFromPosY(local_y, center_y_ratio);
    }, nb::arg("local_y"), nb::arg("center_y_ratio") = 0.5);

mod_imgui.def("push_font",
    [] (ImFont* font) -> void {
        return ImGui::PushFont(font);
    }, nb::arg("font"));

mod_imgui.def("pop_font",
    [] () -> void {
        return ImGui::PopFont();
    });

mod_imgui.def("push_style_color",
    [] (ImGuiCol idx, Color col) -> void {
        return ImGui::PushStyleColor(idx, col);
    }, nb::arg("idx"), nb::arg("col"));

mod_imgui.def("push_style_color_im_vec4",
    [] (ImGuiCol idx, ImVec4 col) -> void {
        return ImGui::PushStyleColor(idx, col);
    }, nb::arg("idx"), nb::arg("col"));

mod_imgui.def("pop_style_color",
    [] (int count) -> void {
        return ImGui::PopStyleColor(count);
    }, nb::arg("count") = 1);

mod_imgui.def("push_style_var",
    [] (ImGuiStyleVar idx, float val) -> void {
        return ImGui::PushStyleVar(idx, val);
    }, nb::arg("idx"), nb::arg("val"));

mod_imgui.def("push_style_var_im_vec2",
    [] (ImGuiStyleVar idx, ImVec2 val) -> void {
        return ImGui::PushStyleVar(idx, val);
    }, nb::arg("idx"), nb::arg("val"));

mod_imgui.def("push_style_var_x",
    [] (ImGuiStyleVar idx, float val_x) -> void {
        return ImGui::PushStyleVarX(idx, val_x);
    }, nb::arg("idx"), nb::arg("val_x"));

mod_imgui.def("push_style_var_y",
    [] (ImGuiStyleVar idx, float val_y) -> void {
        return ImGui::PushStyleVarY(idx, val_y);
    }, nb::arg("idx"), nb::arg("val_y"));

mod_imgui.def("pop_style_var",
    [] (int count) -> void {
        return ImGui::PopStyleVar(count);
    }, nb::arg("count") = 1);

mod_imgui.def("push_item_flag",
    [] (ImGuiItemFlags option, bool enabled) -> void {
        return ImGui::PushItemFlag(option, enabled);
    }, nb::arg("option"), nb::arg("enabled"));

mod_imgui.def("pop_item_flag",
    [] () -> void {
        return ImGui::PopItemFlag();
    });

mod_imgui.def("push_item_width",
    [] (float item_width) -> void {
        return ImGui::PushItemWidth(item_width);
    }, nb::arg("item_width"));

mod_imgui.def("pop_item_width",
    [] () -> void {
        return ImGui::PopItemWidth();
    });

mod_imgui.def("set_next_item_width",
    [] (float item_width) -> void {
        return ImGui::SetNextItemWidth(item_width);
    }, nb::arg("item_width"));

mod_imgui.def("calc_item_width",
    [] () -> float {
        return ImGui::CalcItemWidth();
    });

mod_imgui.def("push_text_wrap_pos",
    [] (float wrap_local_pos_x) -> void {
        return ImGui::PushTextWrapPos(wrap_local_pos_x);
    }, nb::arg("wrap_local_pos_x") = 0.0);

mod_imgui.def("pop_text_wrap_pos",
    [] () -> void {
        return ImGui::PopTextWrapPos();
    });

mod_imgui.def("get_font",
    [] () -> ImFont* {
        return ImGui::GetFont();
    });

mod_imgui.def("get_font_size",
    [] () -> float {
        return ImGui::GetFontSize();
    });

mod_imgui.def("get_font_tex_uv_white_pixel",
    [] () -> ImVec2 {
        return ImGui::GetFontTexUvWhitePixel();
    });

mod_imgui.def("get_color_u32",
    [] (ImGuiCol idx, float alpha_mul) -> Color {
        return ImGui::GetColorU32(idx, alpha_mul);
    }, nb::arg("idx"), nb::arg("alpha_mul") = 1.0);

mod_imgui.def("get_color_u32_im_vec4",
    [] (ImVec4 col) -> Color {
        return ImGui::GetColorU32(col);
    }, nb::arg("col"));

mod_imgui.def("get_color_u32_im_u32",
    [] (Color col, float alpha_mul) -> Color {
        return ImGui::GetColorU32(col, alpha_mul);
    }, nb::arg("col"), nb::arg("alpha_mul") = 1.0);

mod_imgui.def("get_cursor_screen_pos",
    [] () -> ImVec2 {
        return ImGui::GetCursorScreenPos();
    });

mod_imgui.def("set_cursor_screen_pos",
    [] (ImVec2 pos) -> void {
        return ImGui::SetCursorScreenPos(pos);
    }, nb::arg("pos"));

mod_imgui.def("get_content_region_avail",
    [] () -> ImVec2 {
        return ImGui::GetContentRegionAvail();
    });

mod_imgui.def("get_cursor_pos",
    [] () -> ImVec2 {
        return ImGui::GetCursorPos();
    });

mod_imgui.def("get_cursor_pos_x",
    [] () -> float {
        return ImGui::GetCursorPosX();
    });

mod_imgui.def("get_cursor_pos_y",
    [] () -> float {
        return ImGui::GetCursorPosY();
    });

mod_imgui.def("set_cursor_pos",
    [] (ImVec2 local_pos) -> void {
        return ImGui::SetCursorPos(local_pos);
    }, nb::arg("local_pos"));

mod_imgui.def("set_cursor_pos_x",
    [] (float local_x) -> void {
        return ImGui::SetCursorPosX(local_x);
    }, nb::arg("local_x"));

mod_imgui.def("set_cursor_pos_y",
    [] (float local_y) -> void {
        return ImGui::SetCursorPosY(local_y);
    }, nb::arg("local_y"));

mod_imgui.def("get_cursor_start_pos",
    [] () -> ImVec2 {
        return ImGui::GetCursorStartPos();
    });

mod_imgui.def("separator",
    [] () -> void {
        return ImGui::Separator();
    });

mod_imgui.def("same_line",
    [] (float offset_from_start_x, float spacing) -> void {
        return ImGui::SameLine(offset_from_start_x, spacing);
    }, nb::arg("offset_from_start_x") = 0.0, nb::arg("spacing") = -1.0);

mod_imgui.def("new_line",
    [] () -> void {
        return ImGui::NewLine();
    });

mod_imgui.def("spacing",
    [] () -> void {
        return ImGui::Spacing();
    });

mod_imgui.def("dummy",
    [] (ImVec2 size) -> void {
        return ImGui::Dummy(size);
    }, nb::arg("size"));

mod_imgui.def("indent",
    [] (float indent_w) -> void {
        return ImGui::Indent(indent_w);
    }, nb::arg("indent_w") = 0.0);

mod_imgui.def("unindent",
    [] (float indent_w) -> void {
        return ImGui::Unindent(indent_w);
    }, nb::arg("indent_w") = 0.0);

mod_imgui.def("begin_group",
    [] () -> void {
        return ImGui::BeginGroup();
    });

mod_imgui.def("end_group",
    [] () -> void {
        return ImGui::EndGroup();
    });

mod_imgui.def("align_text_to_frame_padding",
    [] () -> void {
        return ImGui::AlignTextToFramePadding();
    });

mod_imgui.def("get_text_line_height",
    [] () -> float {
        return ImGui::GetTextLineHeight();
    });

mod_imgui.def("get_text_line_height_with_spacing",
    [] () -> float {
        return ImGui::GetTextLineHeightWithSpacing();
    });

mod_imgui.def("get_frame_height",
    [] () -> float {
        return ImGui::GetFrameHeight();
    });

mod_imgui.def("get_frame_height_with_spacing",
    [] () -> float {
        return ImGui::GetFrameHeightWithSpacing();
    });

mod_imgui.def("push_id",
    [] (const char* str_id) -> void {
        return ImGui::PushID(str_id);
    }, nb::arg("str_id"));

mod_imgui.def("push_id_str",
    [] (const char* str_id_begin, const char* str_id_end) -> void {
        return ImGui::PushID(str_id_begin, str_id_end);
    }, nb::arg("str_id_begin"), nb::arg("str_id_end"));

mod_imgui.def("push_id_int",
    [] (int int_id) -> void {
        return ImGui::PushID(int_id);
    }, nb::arg("int_id"));

mod_imgui.def("pop_id",
    [] () -> void {
        return ImGui::PopID();
    });

mod_imgui.def("get_id_int",
    [] (int int_id) -> ImGuiID {
        return ImGui::GetID(int_id);
    }, nb::arg("int_id"));

mod_imgui.def("text",
    [] (const char* fmt) -> void {
        return ImGui::Text("%s", fmt);
    }, nb::arg("fmt"));

mod_imgui.def("text_colored",
    [] (ImVec4 col, const char* fmt) -> void {
        return ImGui::TextColored(col, "%s", fmt);
    }, nb::arg("col"), nb::arg("fmt"));

mod_imgui.def("text_disabled",
    [] (const char* fmt) -> void {
        return ImGui::TextDisabled("%s", fmt);
    }, nb::arg("fmt"));

mod_imgui.def("text_wrapped",
    [] (const char* fmt) -> void {
        return ImGui::TextWrapped("%s", fmt);
    }, nb::arg("fmt"));

mod_imgui.def("label_text",
    [] (const char* label, const char* fmt) -> void {
        return ImGui::LabelText(label, "%s", fmt);
    }, nb::arg("label"), nb::arg("fmt"));

mod_imgui.def("bullet_text",
    [] (const char* fmt) -> void {
        return ImGui::BulletText("%s", fmt);
    }, nb::arg("fmt"));

mod_imgui.def("separator_text",
    [] (const char* label) -> void {
        return ImGui::SeparatorText(label);
    }, nb::arg("label"));

mod_imgui.def("button",
    [] (const char* label, ImVec2 size) -> bool {
        return ImGui::Button(label, size);
    }, nb::arg("label"), nb::arg("size") = nb::make_tuple(0, 0));

mod_imgui.def("small_button",
    [] (const char* label) -> bool {
        return ImGui::SmallButton(label);
    }, nb::arg("label"));

mod_imgui.def("invisible_button",
    [] (const char* str_id, ImVec2 size, ImGuiButtonFlags flags) -> bool {
        return ImGui::InvisibleButton(str_id, size, flags);
    }, nb::arg("str_id"), nb::arg("size"), nb::arg("flags") = 0);

mod_imgui.def("arrow_button",
    [] (const char* str_id, ImGuiDir dir) -> bool {
        return ImGui::ArrowButton(str_id, dir);
    }, nb::arg("str_id"), nb::arg("dir"));

mod_imgui.def("checkbox",
    [] (const char* label, bool v) -> nb::tuple {
        auto _call = ImGui::Checkbox(label, &v);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("v"));

mod_imgui.def("radio_button",
    [] (const char* label, bool active) -> bool {
        return ImGui::RadioButton(label, active);
    }, nb::arg("label"), nb::arg("active"));

mod_imgui.def("progress_bar",
    [] (float fraction, ImVec2 size_arg, std::optional<const char*> overlay) -> void {
        return ImGui::ProgressBar(fraction, size_arg, overlay.has_value() ? overlay.value() : NULL);
    }, nb::arg("fraction"), nb::arg("size_arg") = nb::make_tuple(-1.175494351e-38, 0), nb::arg("overlay") = nb::none());

mod_imgui.def("bullet",
    [] () -> void {
        return ImGui::Bullet();
    });

mod_imgui.def("text_link",
    [] (const char* label) -> bool {
        return ImGui::TextLink(label);
    }, nb::arg("label"));

mod_imgui.def("text_link_open_u_r_l",
    [] (const char* label, std::optional<const char*> url) -> void {
        return ImGui::TextLinkOpenURL(label, url.has_value() ? url.value() : NULL);
    }, nb::arg("label"), nb::arg("url") = nb::none());

mod_imgui.def("image",
    [] (ImTextureID user_texture_id, ImVec2 image_size, ImVec2 uv0, ImVec2 uv1) -> void {
        return ImGui::Image(user_texture_id, image_size, uv0, uv1);
    }, nb::arg("user_texture_id"), nb::arg("image_size"), nb::arg("uv0") = nb::make_tuple(0, 0), nb::arg("uv1") = nb::make_tuple(1, 1));

mod_imgui.def("image_with_bg",
    [] (ImTextureID user_texture_id, ImVec2 image_size, ImVec2 uv0, ImVec2 uv1, ImVec4 bg_col, ImVec4 tint_col) -> void {
        return ImGui::ImageWithBg(user_texture_id, image_size, uv0, uv1, bg_col, tint_col);
    }, nb::arg("user_texture_id"), nb::arg("image_size"), nb::arg("uv0") = nb::make_tuple(0, 0), nb::arg("uv1") = nb::make_tuple(1, 1), nb::arg("bg_col") = nb::make_tuple(0, 0, 0, 0), nb::arg("tint_col") = nb::make_tuple(1, 1, 1, 1));

mod_imgui.def("image_button",
    [] (const char* str_id, ImTextureID user_texture_id, ImVec2 image_size, ImVec2 uv0, ImVec2 uv1, ImVec4 bg_col, ImVec4 tint_col) -> bool {
        return ImGui::ImageButton(str_id, user_texture_id, image_size, uv0, uv1, bg_col, tint_col);
    }, nb::arg("str_id"), nb::arg("user_texture_id"), nb::arg("image_size"), nb::arg("uv0") = nb::make_tuple(0, 0), nb::arg("uv1") = nb::make_tuple(1, 1), nb::arg("bg_col") = nb::make_tuple(0, 0, 0, 0), nb::arg("tint_col") = nb::make_tuple(1, 1, 1, 1));

mod_imgui.def("drag_float",
    [] (const char* label, float v, float v_speed, float v_min, float v_max, const char* format, ImGuiSliderFlags flags) -> nb::tuple {
        auto _call = ImGui::DragFloat(label, &v, v_speed, v_min, v_max, format, flags);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("v"), nb::arg("v_speed") = 1.0, nb::arg("v_min") = 0.0, nb::arg("v_max") = 0.0, nb::arg("format") = "%.3f", nb::arg("flags") = 0);

mod_imgui.def("drag_float2",
    [] (const char* label, std::array<float, 2> v, float v_speed, float v_min, float v_max, const char* format, ImGuiSliderFlags flags) -> nb::tuple {
        auto _call = ImGui::DragFloat2(label, v.data(), v_speed, v_min, v_max, format, flags);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("v"), nb::arg("v_speed") = 1.0, nb::arg("v_min") = 0.0, nb::arg("v_max") = 0.0, nb::arg("format") = "%.3f", nb::arg("flags") = 0);

mod_imgui.def("drag_float3",
    [] (const char* label, std::array<float, 3> v, float v_speed, float v_min, float v_max, const char* format, ImGuiSliderFlags flags) -> nb::tuple {
        auto _call = ImGui::DragFloat3(label, v.data(), v_speed, v_min, v_max, format, flags);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("v"), nb::arg("v_speed") = 1.0, nb::arg("v_min") = 0.0, nb::arg("v_max") = 0.0, nb::arg("format") = "%.3f", nb::arg("flags") = 0);

mod_imgui.def("drag_float4",
    [] (const char* label, std::array<float, 4> v, float v_speed, float v_min, float v_max, const char* format, ImGuiSliderFlags flags) -> nb::tuple {
        auto _call = ImGui::DragFloat4(label, v.data(), v_speed, v_min, v_max, format, flags);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("v"), nb::arg("v_speed") = 1.0, nb::arg("v_min") = 0.0, nb::arg("v_max") = 0.0, nb::arg("format") = "%.3f", nb::arg("flags") = 0);

mod_imgui.def("drag_int",
    [] (const char* label, int v, float v_speed, int v_min, int v_max, const char* format, ImGuiSliderFlags flags) -> nb::tuple {
        auto _call = ImGui::DragInt(label, &v, v_speed, v_min, v_max, format, flags);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("v"), nb::arg("v_speed") = 1.0, nb::arg("v_min") = 0, nb::arg("v_max") = 0, nb::arg("format") = "%d", nb::arg("flags") = 0);

mod_imgui.def("drag_int2",
    [] (const char* label, std::array<int, 2> v, float v_speed, int v_min, int v_max, const char* format, ImGuiSliderFlags flags) -> nb::tuple {
        auto _call = ImGui::DragInt2(label, v.data(), v_speed, v_min, v_max, format, flags);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("v"), nb::arg("v_speed") = 1.0, nb::arg("v_min") = 0, nb::arg("v_max") = 0, nb::arg("format") = "%d", nb::arg("flags") = 0);

mod_imgui.def("drag_int3",
    [] (const char* label, std::array<int, 3> v, float v_speed, int v_min, int v_max, const char* format, ImGuiSliderFlags flags) -> nb::tuple {
        auto _call = ImGui::DragInt3(label, v.data(), v_speed, v_min, v_max, format, flags);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("v"), nb::arg("v_speed") = 1.0, nb::arg("v_min") = 0, nb::arg("v_max") = 0, nb::arg("format") = "%d", nb::arg("flags") = 0);

mod_imgui.def("drag_int4",
    [] (const char* label, std::array<int, 4> v, float v_speed, int v_min, int v_max, const char* format, ImGuiSliderFlags flags) -> nb::tuple {
        auto _call = ImGui::DragInt4(label, v.data(), v_speed, v_min, v_max, format, flags);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("v"), nb::arg("v_speed") = 1.0, nb::arg("v_min") = 0, nb::arg("v_max") = 0, nb::arg("format") = "%d", nb::arg("flags") = 0);

mod_imgui.def("slider_float",
    [] (const char* label, float v, float v_min, float v_max, const char* format, ImGuiSliderFlags flags) -> nb::tuple {
        auto _call = ImGui::SliderFloat(label, &v, v_min, v_max, format, flags);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("v"), nb::arg("v_min"), nb::arg("v_max"), nb::arg("format") = "%.3f", nb::arg("flags") = 0);

mod_imgui.def("slider_float2",
    [] (const char* label, std::array<float, 2> v, float v_min, float v_max, const char* format, ImGuiSliderFlags flags) -> nb::tuple {
        auto _call = ImGui::SliderFloat2(label, v.data(), v_min, v_max, format, flags);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("v"), nb::arg("v_min"), nb::arg("v_max"), nb::arg("format") = "%.3f", nb::arg("flags") = 0);

mod_imgui.def("slider_float3",
    [] (const char* label, std::array<float, 3> v, float v_min, float v_max, const char* format, ImGuiSliderFlags flags) -> nb::tuple {
        auto _call = ImGui::SliderFloat3(label, v.data(), v_min, v_max, format, flags);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("v"), nb::arg("v_min"), nb::arg("v_max"), nb::arg("format") = "%.3f", nb::arg("flags") = 0);

mod_imgui.def("slider_float4",
    [] (const char* label, std::array<float, 4> v, float v_min, float v_max, const char* format, ImGuiSliderFlags flags) -> nb::tuple {
        auto _call = ImGui::SliderFloat4(label, v.data(), v_min, v_max, format, flags);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("v"), nb::arg("v_min"), nb::arg("v_max"), nb::arg("format") = "%.3f", nb::arg("flags") = 0);

mod_imgui.def("slider_angle",
    [] (const char* label, float v_rad, float v_degrees_min, float v_degrees_max, const char* format, ImGuiSliderFlags flags) -> nb::tuple {
        auto _call = ImGui::SliderAngle(label, &v_rad, v_degrees_min, v_degrees_max, format, flags);
        auto _ret = v_rad;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("v_rad"), nb::arg("v_degrees_min") = -360.0, nb::arg("v_degrees_max") = +360.0, nb::arg("format") = "%.0f deg", nb::arg("flags") = 0);

mod_imgui.def("slider_int",
    [] (const char* label, int v, int v_min, int v_max, const char* format, ImGuiSliderFlags flags) -> nb::tuple {
        auto _call = ImGui::SliderInt(label, &v, v_min, v_max, format, flags);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("v"), nb::arg("v_min"), nb::arg("v_max"), nb::arg("format") = "%d", nb::arg("flags") = 0);

mod_imgui.def("slider_int2",
    [] (const char* label, std::array<int, 2> v, int v_min, int v_max, const char* format, ImGuiSliderFlags flags) -> nb::tuple {
        auto _call = ImGui::SliderInt2(label, v.data(), v_min, v_max, format, flags);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("v"), nb::arg("v_min"), nb::arg("v_max"), nb::arg("format") = "%d", nb::arg("flags") = 0);

mod_imgui.def("slider_int3",
    [] (const char* label, std::array<int, 3> v, int v_min, int v_max, const char* format, ImGuiSliderFlags flags) -> nb::tuple {
        auto _call = ImGui::SliderInt3(label, v.data(), v_min, v_max, format, flags);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("v"), nb::arg("v_min"), nb::arg("v_max"), nb::arg("format") = "%d", nb::arg("flags") = 0);

mod_imgui.def("slider_int4",
    [] (const char* label, std::array<int, 4> v, int v_min, int v_max, const char* format, ImGuiSliderFlags flags) -> nb::tuple {
        auto _call = ImGui::SliderInt4(label, v.data(), v_min, v_max, format, flags);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("v"), nb::arg("v_min"), nb::arg("v_max"), nb::arg("format") = "%d", nb::arg("flags") = 0);

mod_imgui.def("v_slider_float",
    [] (const char* label, ImVec2 size, float v, float v_min, float v_max, const char* format, ImGuiSliderFlags flags) -> nb::tuple {
        auto _call = ImGui::VSliderFloat(label, size, &v, v_min, v_max, format, flags);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("size"), nb::arg("v"), nb::arg("v_min"), nb::arg("v_max"), nb::arg("format") = "%.3f", nb::arg("flags") = 0);

mod_imgui.def("v_slider_int",
    [] (const char* label, ImVec2 size, int v, int v_min, int v_max, const char* format, ImGuiSliderFlags flags) -> nb::tuple {
        auto _call = ImGui::VSliderInt(label, size, &v, v_min, v_max, format, flags);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("size"), nb::arg("v"), nb::arg("v_min"), nb::arg("v_max"), nb::arg("format") = "%d", nb::arg("flags") = 0);

mod_imgui.def("input_float",
    [] (const char* label, float v, float step, float step_fast, const char* format, ImGuiInputTextFlags flags) -> nb::tuple {
        auto _call = ImGui::InputFloat(label, &v, step, step_fast, format, flags);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("v"), nb::arg("step") = 0.0, nb::arg("step_fast") = 0.0, nb::arg("format") = "%.3f", nb::arg("flags") = 0);

mod_imgui.def("input_float2",
    [] (const char* label, std::array<float, 2> v, const char* format, ImGuiInputTextFlags flags) -> nb::tuple {
        auto _call = ImGui::InputFloat2(label, v.data(), format, flags);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("v"), nb::arg("format") = "%.3f", nb::arg("flags") = 0);

mod_imgui.def("input_float3",
    [] (const char* label, std::array<float, 3> v, const char* format, ImGuiInputTextFlags flags) -> nb::tuple {
        auto _call = ImGui::InputFloat3(label, v.data(), format, flags);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("v"), nb::arg("format") = "%.3f", nb::arg("flags") = 0);

mod_imgui.def("input_float4",
    [] (const char* label, std::array<float, 4> v, const char* format, ImGuiInputTextFlags flags) -> nb::tuple {
        auto _call = ImGui::InputFloat4(label, v.data(), format, flags);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("v"), nb::arg("format") = "%.3f", nb::arg("flags") = 0);

mod_imgui.def("input_int",
    [] (const char* label, int v, int step, int step_fast, ImGuiInputTextFlags flags) -> nb::tuple {
        auto _call = ImGui::InputInt(label, &v, step, step_fast, flags);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("v"), nb::arg("step") = 1, nb::arg("step_fast") = 100, nb::arg("flags") = 0);

mod_imgui.def("input_int2",
    [] (const char* label, std::array<int, 2> v, ImGuiInputTextFlags flags) -> nb::tuple {
        auto _call = ImGui::InputInt2(label, v.data(), flags);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("v"), nb::arg("flags") = 0);

mod_imgui.def("input_int3",
    [] (const char* label, std::array<int, 3> v, ImGuiInputTextFlags flags) -> nb::tuple {
        auto _call = ImGui::InputInt3(label, v.data(), flags);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("v"), nb::arg("flags") = 0);

mod_imgui.def("input_int4",
    [] (const char* label, std::array<int, 4> v, ImGuiInputTextFlags flags) -> nb::tuple {
        auto _call = ImGui::InputInt4(label, v.data(), flags);
        auto _ret = v;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("v"), nb::arg("flags") = 0);

mod_imgui.def("color_edit3",
    [] (const char* label, std::array<float, 3> col, ImGuiColorEditFlags flags) -> nb::tuple {
        auto _call = ImGui::ColorEdit3(label, col.data(), flags);
        auto _ret = col;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("col"), nb::arg("flags") = 0);

mod_imgui.def("color_edit4",
    [] (const char* label, std::array<float, 4> col, ImGuiColorEditFlags flags) -> nb::tuple {
        auto _call = ImGui::ColorEdit4(label, col.data(), flags);
        auto _ret = col;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("col"), nb::arg("flags") = 0);

mod_imgui.def("color_picker3",
    [] (const char* label, std::array<float, 3> col, ImGuiColorEditFlags flags) -> nb::tuple {
        auto _call = ImGui::ColorPicker3(label, col.data(), flags);
        auto _ret = col;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("col"), nb::arg("flags") = 0);

mod_imgui.def("color_button",
    [] (const char* desc_id, ImVec4 col, ImGuiColorEditFlags flags, ImVec2 size) -> bool {
        return ImGui::ColorButton(desc_id, col, flags, size);
    }, nb::arg("desc_id"), nb::arg("col"), nb::arg("flags") = 0, nb::arg("size") = nb::make_tuple(0, 0));

mod_imgui.def("set_color_edit_options",
    [] (ImGuiColorEditFlags flags) -> void {
        return ImGui::SetColorEditOptions(flags);
    }, nb::arg("flags"));

mod_imgui.def("tree_node",
    [] (const char* label) -> bool {
        return ImGui::TreeNode(label);
    }, nb::arg("label"));

mod_imgui.def("tree_node_str",
    [] (const char* str_id, const char* fmt) -> bool {
        return ImGui::TreeNode(str_id, "%s", fmt);
    }, nb::arg("str_id"), nb::arg("fmt"));

mod_imgui.def("tree_node_ex",
    [] (const char* label, ImGuiTreeNodeFlags flags) -> bool {
        return ImGui::TreeNodeEx(label, flags);
    }, nb::arg("label"), nb::arg("flags") = 0);

mod_imgui.def("tree_node_ex_str",
    [] (const char* str_id, ImGuiTreeNodeFlags flags, const char* fmt) -> bool {
        return ImGui::TreeNodeEx(str_id, flags, "%s", fmt);
    }, nb::arg("str_id"), nb::arg("flags"), nb::arg("fmt"));

mod_imgui.def("tree_push",
    [] (const char* str_id) -> void {
        return ImGui::TreePush(str_id);
    }, nb::arg("str_id"));

mod_imgui.def("tree_pop",
    [] () -> void {
        return ImGui::TreePop();
    });

mod_imgui.def("get_tree_node_to_label_spacing",
    [] () -> float {
        return ImGui::GetTreeNodeToLabelSpacing();
    });

mod_imgui.def("collapsing_header",
    [] (const char* label, ImGuiTreeNodeFlags flags) -> bool {
        return ImGui::CollapsingHeader(label, flags);
    }, nb::arg("label"), nb::arg("flags") = 0);

mod_imgui.def("set_next_item_open",
    [] (bool is_open, ImGuiCond cond) -> void {
        return ImGui::SetNextItemOpen(is_open, cond);
    }, nb::arg("is_open"), nb::arg("cond") = 0);

mod_imgui.def("set_next_item_storage_id",
    [] (ImGuiID storage_id) -> void {
        return ImGui::SetNextItemStorageID(storage_id);
    }, nb::arg("storage_id"));

mod_imgui.def("selectable",
    [] (const char* label, bool selected, ImGuiSelectableFlags flags, ImVec2 size) -> bool {
        return ImGui::Selectable(label, selected, flags, size);
    }, nb::arg("label"), nb::arg("selected") = false, nb::arg("flags") = 0, nb::arg("size") = nb::make_tuple(0, 0));

mod_imgui.def("begin_multi_select",
    [] (ImGuiMultiSelectFlags flags, int selection_size, int items_count) -> ImGuiMultiSelectIO* {
        return ImGui::BeginMultiSelect(flags, selection_size, items_count);
    }, nb::arg("flags"), nb::arg("selection_size") = -1, nb::arg("items_count") = -1);

mod_imgui.def("end_multi_select",
    [] () -> ImGuiMultiSelectIO* {
        return ImGui::EndMultiSelect();
    });

mod_imgui.def("set_next_item_selection_user_data",
    [] (ImGuiSelectionUserData selection_user_data) -> void {
        return ImGui::SetNextItemSelectionUserData(selection_user_data);
    }, nb::arg("selection_user_data"));

mod_imgui.def("is_item_toggled_selection",
    [] () -> bool {
        return ImGui::IsItemToggledSelection();
    });

mod_imgui.def("begin_menu_bar",
    [] () -> bool {
        return ImGui::BeginMenuBar();
    });

mod_imgui.def("end_menu_bar",
    [] () -> void {
        return ImGui::EndMenuBar();
    });

mod_imgui.def("begin_main_menu_bar",
    [] () -> bool {
        return ImGui::BeginMainMenuBar();
    });

mod_imgui.def("end_main_menu_bar",
    [] () -> void {
        return ImGui::EndMainMenuBar();
    });

mod_imgui.def("begin_menu",
    [] (const char* label, bool enabled) -> bool {
        return ImGui::BeginMenu(label, enabled);
    }, nb::arg("label"), nb::arg("enabled") = true);

mod_imgui.def("end_menu",
    [] () -> void {
        return ImGui::EndMenu();
    });

mod_imgui.def("menu_item",
    [] (const char* label, std::optional<const char*> shortcut, bool selected, bool enabled) -> bool {
        return ImGui::MenuItem(label, shortcut.has_value() ? shortcut.value() : NULL, selected, enabled);
    }, nb::arg("label"), nb::arg("shortcut") = nb::none(), nb::arg("selected") = false, nb::arg("enabled") = true);

mod_imgui.def("begin_tooltip",
    [] () -> bool {
        return ImGui::BeginTooltip();
    });

mod_imgui.def("end_tooltip",
    [] () -> void {
        return ImGui::EndTooltip();
    });

mod_imgui.def("set_tooltip",
    [] (const char* fmt) -> void {
        return ImGui::SetTooltip("%s", fmt);
    }, nb::arg("fmt"));

mod_imgui.def("begin_item_tooltip",
    [] () -> bool {
        return ImGui::BeginItemTooltip();
    });

mod_imgui.def("set_item_tooltip",
    [] (const char* fmt) -> void {
        return ImGui::SetItemTooltip("%s", fmt);
    }, nb::arg("fmt"));

mod_imgui.def("begin_popup",
    [] (const char* str_id, ImGuiWindowFlags flags) -> bool {
        return ImGui::BeginPopup(str_id, flags);
    }, nb::arg("str_id"), nb::arg("flags") = 0);

mod_imgui.def("begin_popup_modal",
    [] (const char* name, std::optional<bool> p_open, ImGuiWindowFlags flags) -> nb::tuple {
        auto _call = ImGui::BeginPopupModal(name, p_open.has_value() ? &p_open.value() : NULL, flags);
        auto _ret = p_open;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("name"), nb::arg("p_open") = nb::none(), nb::arg("flags") = 0);

mod_imgui.def("end_popup",
    [] () -> void {
        return ImGui::EndPopup();
    });

mod_imgui.def("open_popup",
    [] (const char* str_id, ImGuiPopupFlags popup_flags) -> void {
        return ImGui::OpenPopup(str_id, popup_flags);
    }, nb::arg("str_id"), nb::arg("popup_flags") = 0);

mod_imgui.def("open_popup_on_item_click",
    [] (std::optional<const char*> str_id, ImGuiPopupFlags popup_flags) -> void {
        return ImGui::OpenPopupOnItemClick(str_id.has_value() ? str_id.value() : NULL, popup_flags);
    }, nb::arg("str_id") = nb::none(), nb::arg("popup_flags") = 1);

mod_imgui.def("close_current_popup",
    [] () -> void {
        return ImGui::CloseCurrentPopup();
    });

mod_imgui.def("begin_popup_context_item",
    [] (std::optional<const char*> str_id, ImGuiPopupFlags popup_flags) -> bool {
        return ImGui::BeginPopupContextItem(str_id.has_value() ? str_id.value() : NULL, popup_flags);
    }, nb::arg("str_id") = nb::none(), nb::arg("popup_flags") = 1);

mod_imgui.def("begin_popup_context_window",
    [] (std::optional<const char*> str_id, ImGuiPopupFlags popup_flags) -> bool {
        return ImGui::BeginPopupContextWindow(str_id.has_value() ? str_id.value() : NULL, popup_flags);
    }, nb::arg("str_id") = nb::none(), nb::arg("popup_flags") = 1);

mod_imgui.def("begin_popup_context_void",
    [] (std::optional<const char*> str_id, ImGuiPopupFlags popup_flags) -> bool {
        return ImGui::BeginPopupContextVoid(str_id.has_value() ? str_id.value() : NULL, popup_flags);
    }, nb::arg("str_id") = nb::none(), nb::arg("popup_flags") = 1);

mod_imgui.def("is_popup_open",
    [] (const char* str_id, ImGuiPopupFlags flags) -> bool {
        return ImGui::IsPopupOpen(str_id, flags);
    }, nb::arg("str_id"), nb::arg("flags") = 0);

mod_imgui.def("begin_table",
    [] (const char* str_id, int columns, ImGuiTableFlags flags, ImVec2 outer_size, float inner_width) -> bool {
        return ImGui::BeginTable(str_id, columns, flags, outer_size, inner_width);
    }, nb::arg("str_id"), nb::arg("columns"), nb::arg("flags") = 0, nb::arg("outer_size") = nb::make_tuple(0.0, 0.0), nb::arg("inner_width") = 0.0);

mod_imgui.def("end_table",
    [] () -> void {
        return ImGui::EndTable();
    });

mod_imgui.def("table_next_row",
    [] (ImGuiTableRowFlags row_flags, float min_row_height) -> void {
        return ImGui::TableNextRow(row_flags, min_row_height);
    }, nb::arg("row_flags") = 0, nb::arg("min_row_height") = 0.0);

mod_imgui.def("table_next_column",
    [] () -> bool {
        return ImGui::TableNextColumn();
    });

mod_imgui.def("table_set_column_index",
    [] (int column_n) -> bool {
        return ImGui::TableSetColumnIndex(column_n);
    }, nb::arg("column_n"));

mod_imgui.def("table_setup_scroll_freeze",
    [] (int cols, int rows) -> void {
        return ImGui::TableSetupScrollFreeze(cols, rows);
    }, nb::arg("cols"), nb::arg("rows"));

mod_imgui.def("table_header",
    [] (const char* label) -> void {
        return ImGui::TableHeader(label);
    }, nb::arg("label"));

mod_imgui.def("table_headers_row",
    [] () -> void {
        return ImGui::TableHeadersRow();
    });

mod_imgui.def("table_angled_headers_row",
    [] () -> void {
        return ImGui::TableAngledHeadersRow();
    });

mod_imgui.def("table_get_column_count",
    [] () -> int {
        return ImGui::TableGetColumnCount();
    });

mod_imgui.def("table_get_column_index",
    [] () -> int {
        return ImGui::TableGetColumnIndex();
    });

mod_imgui.def("table_get_row_index",
    [] () -> int {
        return ImGui::TableGetRowIndex();
    });

mod_imgui.def("table_get_column_name",
    [] (int column_n) -> const char* {
        return ImGui::TableGetColumnName(column_n);
    }, nb::arg("column_n") = -1);

mod_imgui.def("table_get_column_flags",
    [] (int column_n) -> ImGuiTableColumnFlags {
        return ImGui::TableGetColumnFlags(column_n);
    }, nb::arg("column_n") = -1);

mod_imgui.def("table_set_column_enabled",
    [] (int column_n, bool v) -> void {
        return ImGui::TableSetColumnEnabled(column_n, v);
    }, nb::arg("column_n"), nb::arg("v"));

mod_imgui.def("table_get_hovered_column",
    [] () -> int {
        return ImGui::TableGetHoveredColumn();
    });

mod_imgui.def("table_set_bg_color",
    [] (ImGuiTableBgTarget target, Color color, int column_n) -> void {
        return ImGui::TableSetBgColor(target, color, column_n);
    }, nb::arg("target"), nb::arg("color"), nb::arg("column_n") = -1);

mod_imgui.def("columns",
    [] (int count, std::optional<const char*> id, bool borders) -> void {
        return ImGui::Columns(count, id.has_value() ? id.value() : NULL, borders);
    }, nb::arg("count") = 1, nb::arg("id") = nb::none(), nb::arg("borders") = true);

mod_imgui.def("next_column",
    [] () -> void {
        return ImGui::NextColumn();
    });

mod_imgui.def("get_column_index",
    [] () -> int {
        return ImGui::GetColumnIndex();
    });

mod_imgui.def("get_column_width",
    [] (int column_index) -> float {
        return ImGui::GetColumnWidth(column_index);
    }, nb::arg("column_index") = -1);

mod_imgui.def("set_column_width",
    [] (int column_index, float width) -> void {
        return ImGui::SetColumnWidth(column_index, width);
    }, nb::arg("column_index"), nb::arg("width"));

mod_imgui.def("get_column_offset",
    [] (int column_index) -> float {
        return ImGui::GetColumnOffset(column_index);
    }, nb::arg("column_index") = -1);

mod_imgui.def("set_column_offset",
    [] (int column_index, float offset_x) -> void {
        return ImGui::SetColumnOffset(column_index, offset_x);
    }, nb::arg("column_index"), nb::arg("offset_x"));

mod_imgui.def("get_columns_count",
    [] () -> int {
        return ImGui::GetColumnsCount();
    });

mod_imgui.def("begin_tab_bar",
    [] (const char* str_id, ImGuiTabBarFlags flags) -> bool {
        return ImGui::BeginTabBar(str_id, flags);
    }, nb::arg("str_id"), nb::arg("flags") = 0);

mod_imgui.def("end_tab_bar",
    [] () -> void {
        return ImGui::EndTabBar();
    });

mod_imgui.def("begin_tab_item",
    [] (const char* label, std::optional<bool> p_open, ImGuiTabItemFlags flags) -> nb::tuple {
        auto _call = ImGui::BeginTabItem(label, p_open.has_value() ? &p_open.value() : NULL, flags);
        auto _ret = p_open;
        return nb::make_tuple(_call, _ret);
    }, nb::arg("label"), nb::arg("p_open") = nb::none(), nb::arg("flags") = 0);

mod_imgui.def("end_tab_item",
    [] () -> void {
        return ImGui::EndTabItem();
    });

mod_imgui.def("tab_item_button",
    [] (const char* label, ImGuiTabItemFlags flags) -> bool {
        return ImGui::TabItemButton(label, flags);
    }, nb::arg("label"), nb::arg("flags") = 0);

mod_imgui.def("set_tab_item_closed",
    [] (const char* tab_or_docked_window_label) -> void {
        return ImGui::SetTabItemClosed(tab_or_docked_window_label);
    }, nb::arg("tab_or_docked_window_label"));

mod_imgui.def("dock_space",
    [] (ImGuiID dockspace_id, ImVec2 size, ImGuiDockNodeFlags flags, std::optional<const ImGuiWindowClass*> window_class) -> ImGuiID {
        return ImGui::DockSpace(dockspace_id, size, flags, window_class.has_value() ? window_class.value() : NULL);
    }, nb::arg("dockspace_id"), nb::arg("size") = nb::make_tuple(0, 0), nb::arg("flags") = 0, nb::arg("window_class") = nb::none());

mod_imgui.def("dock_space_over_viewport",
    [] (ImGuiID dockspace_id, std::optional<const ImGuiViewport*> viewport, ImGuiDockNodeFlags flags, std::optional<const ImGuiWindowClass*> window_class) -> ImGuiID {
        return ImGui::DockSpaceOverViewport(dockspace_id, viewport.has_value() ? viewport.value() : NULL, flags, window_class.has_value() ? window_class.value() : NULL);
    }, nb::arg("dockspace_id") = 0, nb::arg("viewport") = nb::none(), nb::arg("flags") = 0, nb::arg("window_class") = nb::none());

mod_imgui.def("set_next_window_dock_id",
    [] (ImGuiID dock_id, ImGuiCond cond) -> void {
        return ImGui::SetNextWindowDockID(dock_id, cond);
    }, nb::arg("dock_id"), nb::arg("cond") = 0);

mod_imgui.def("set_next_window_class",
    [] (const ImGuiWindowClass* window_class) -> void {
        return ImGui::SetNextWindowClass(window_class);
    }, nb::arg("window_class"));

mod_imgui.def("get_window_dock_id",
    [] () -> ImGuiID {
        return ImGui::GetWindowDockID();
    });

mod_imgui.def("is_window_docked",
    [] () -> bool {
        return ImGui::IsWindowDocked();
    });

mod_imgui.def("log_to_t_t_y",
    [] (int auto_open_depth) -> void {
        return ImGui::LogToTTY(auto_open_depth);
    }, nb::arg("auto_open_depth") = -1);

mod_imgui.def("log_to_file",
    [] (int auto_open_depth, std::optional<const char*> filename) -> void {
        return ImGui::LogToFile(auto_open_depth, filename.has_value() ? filename.value() : NULL);
    }, nb::arg("auto_open_depth") = -1, nb::arg("filename") = nb::none());

mod_imgui.def("log_to_clipboard",
    [] (int auto_open_depth) -> void {
        return ImGui::LogToClipboard(auto_open_depth);
    }, nb::arg("auto_open_depth") = -1);

mod_imgui.def("log_finish",
    [] () -> void {
        return ImGui::LogFinish();
    });

mod_imgui.def("log_buttons",
    [] () -> void {
        return ImGui::LogButtons();
    });

mod_imgui.def("log_text",
    [] (const char* fmt) -> void {
        return ImGui::LogText("%s", fmt);
    }, nb::arg("fmt"));

mod_imgui.def("begin_drag_drop_source",
    [] (ImGuiDragDropFlags flags) -> bool {
        return ImGui::BeginDragDropSource(flags);
    }, nb::arg("flags") = 0);

mod_imgui.def("set_drag_drop_payload",
    [] (const char* type, std::vector<uint8_t> data, size_t sz, ImGuiCond cond) -> bool {
        return ImGui::SetDragDropPayload(type, data.data(), sz, cond);
    }, nb::arg("type"), nb::arg("data"), nb::arg("sz"), nb::arg("cond") = 0);

mod_imgui.def("end_drag_drop_source",
    [] () -> void {
        return ImGui::EndDragDropSource();
    });

mod_imgui.def("begin_drag_drop_target",
    [] () -> bool {
        return ImGui::BeginDragDropTarget();
    });

mod_imgui.def("accept_drag_drop_payload",
    [] (const char* type, ImGuiDragDropFlags flags) -> const ImGuiPayload* {
        return ImGui::AcceptDragDropPayload(type, flags);
    }, nb::arg("type"), nb::arg("flags") = 0);

mod_imgui.def("end_drag_drop_target",
    [] () -> void {
        return ImGui::EndDragDropTarget();
    });

mod_imgui.def("get_drag_drop_payload",
    [] () -> const ImGuiPayload* {
        return ImGui::GetDragDropPayload();
    });

mod_imgui.def("begin_disabled",
    [] (bool disabled) -> void {
        return ImGui::BeginDisabled(disabled);
    }, nb::arg("disabled") = true);

mod_imgui.def("end_disabled",
    [] () -> void {
        return ImGui::EndDisabled();
    });

mod_imgui.def("push_clip_rect",
    [] (ImVec2 clip_rect_min, ImVec2 clip_rect_max, bool intersect_with_current_clip_rect) -> void {
        return ImGui::PushClipRect(clip_rect_min, clip_rect_max, intersect_with_current_clip_rect);
    }, nb::arg("clip_rect_min"), nb::arg("clip_rect_max"), nb::arg("intersect_with_current_clip_rect"));

mod_imgui.def("pop_clip_rect",
    [] () -> void {
        return ImGui::PopClipRect();
    });

mod_imgui.def("set_item_default_focus",
    [] () -> void {
        return ImGui::SetItemDefaultFocus();
    });

mod_imgui.def("set_keyboard_focus_here",
    [] (int offset) -> void {
        return ImGui::SetKeyboardFocusHere(offset);
    }, nb::arg("offset") = 0);

mod_imgui.def("set_nav_cursor_visible",
    [] (bool visible) -> void {
        return ImGui::SetNavCursorVisible(visible);
    }, nb::arg("visible"));

mod_imgui.def("set_next_item_allow_overlap",
    [] () -> void {
        return ImGui::SetNextItemAllowOverlap();
    });

mod_imgui.def("is_item_hovered",
    [] (ImGuiHoveredFlags flags) -> bool {
        return ImGui::IsItemHovered(flags);
    }, nb::arg("flags") = 0);

mod_imgui.def("is_item_active",
    [] () -> bool {
        return ImGui::IsItemActive();
    });

mod_imgui.def("is_item_focused",
    [] () -> bool {
        return ImGui::IsItemFocused();
    });

mod_imgui.def("is_item_clicked",
    [] (ImGuiMouseButton mouse_button) -> bool {
        return ImGui::IsItemClicked(mouse_button);
    }, nb::arg("mouse_button") = 0);

mod_imgui.def("is_item_visible",
    [] () -> bool {
        return ImGui::IsItemVisible();
    });

mod_imgui.def("is_item_edited",
    [] () -> bool {
        return ImGui::IsItemEdited();
    });

mod_imgui.def("is_item_activated",
    [] () -> bool {
        return ImGui::IsItemActivated();
    });

mod_imgui.def("is_item_deactivated",
    [] () -> bool {
        return ImGui::IsItemDeactivated();
    });

mod_imgui.def("is_item_deactivated_after_edit",
    [] () -> bool {
        return ImGui::IsItemDeactivatedAfterEdit();
    });

mod_imgui.def("is_item_toggled_open",
    [] () -> bool {
        return ImGui::IsItemToggledOpen();
    });

mod_imgui.def("is_any_item_hovered",
    [] () -> bool {
        return ImGui::IsAnyItemHovered();
    });

mod_imgui.def("is_any_item_active",
    [] () -> bool {
        return ImGui::IsAnyItemActive();
    });

mod_imgui.def("is_any_item_focused",
    [] () -> bool {
        return ImGui::IsAnyItemFocused();
    });

mod_imgui.def("get_item_id",
    [] () -> ImGuiID {
        return ImGui::GetItemID();
    });

mod_imgui.def("get_item_rect_min",
    [] () -> ImVec2 {
        return ImGui::GetItemRectMin();
    });

mod_imgui.def("get_item_rect_max",
    [] () -> ImVec2 {
        return ImGui::GetItemRectMax();
    });

mod_imgui.def("get_item_rect_size",
    [] () -> ImVec2 {
        return ImGui::GetItemRectSize();
    });

mod_imgui.def("get_main_viewport",
    [] () -> ImGuiViewport* {
        return ImGui::GetMainViewport();
    });

mod_imgui.def("get_background_draw_list",
    [] (std::optional<ImGuiViewport*> viewport) -> DrawList* {
        return new DrawList(ImGui::GetBackgroundDrawList(viewport.has_value() ? viewport.value() : NULL));
    }, nb::arg("viewport") = nb::none());

mod_imgui.def("get_foreground_draw_list",
    [] (std::optional<ImGuiViewport*> viewport) -> DrawList* {
        return new DrawList(ImGui::GetForegroundDrawList(viewport.has_value() ? viewport.value() : NULL));
    }, nb::arg("viewport") = nb::none());

mod_imgui.def("is_rect_visible_by_size",
    [] (ImVec2 size) -> bool {
        return ImGui::IsRectVisible(size);
    }, nb::arg("size"));

mod_imgui.def("is_rect_visible",
    [] (ImVec2 rect_min, ImVec2 rect_max) -> bool {
        return ImGui::IsRectVisible(rect_min, rect_max);
    }, nb::arg("rect_min"), nb::arg("rect_max"));

mod_imgui.def("get_time",
    [] () -> double {
        return ImGui::GetTime();
    });

mod_imgui.def("get_frame_count",
    [] () -> int {
        return ImGui::GetFrameCount();
    });

mod_imgui.def("get_style_color_name",
    [] (ImGuiCol idx) -> const char* {
        return ImGui::GetStyleColorName(idx);
    }, nb::arg("idx"));

mod_imgui.def("set_state_storage",
    [] (ImGuiStorage* storage) -> void {
        return ImGui::SetStateStorage(storage);
    }, nb::arg("storage"));

mod_imgui.def("get_state_storage",
    [] () -> ImGuiStorage* {
        return ImGui::GetStateStorage();
    });

mod_imgui.def("calc_text_size",
    [] (const char* text, bool hide_text_after_double_hash, float wrap_width) -> ImVec2 {
        return ImGui::CalcTextSize(text, nullptr, hide_text_after_double_hash, wrap_width);
    }, nb::arg("text"), nb::arg("hide_text_after_double_hash") = false, nb::arg("wrap_width") = -1.0);

mod_imgui.def("color_convert_u32_to_float4",
    [] (Color value) -> ImVec4 {
        return ImGui::ColorConvertU32ToFloat4(value);
    }, nb::arg("value"));

mod_imgui.def("color_convert_float4_to_u32",
    [] (ImVec4 value) -> Color {
        return ImGui::ColorConvertFloat4ToU32(value);
    }, nb::arg("value"));

mod_imgui.def("is_key_down",
    [] (ImGuiKey key) -> bool {
        return ImGui::IsKeyDown(key);
    }, nb::arg("key"));

mod_imgui.def("is_key_pressed",
    [] (ImGuiKey key, bool repeat) -> bool {
        return ImGui::IsKeyPressed(key, repeat);
    }, nb::arg("key"), nb::arg("repeat") = true);

mod_imgui.def("is_key_released",
    [] (ImGuiKey key) -> bool {
        return ImGui::IsKeyReleased(key);
    }, nb::arg("key"));

mod_imgui.def("is_key_chord_pressed",
    [] (ImGuiKeyChord key_chord) -> bool {
        return ImGui::IsKeyChordPressed(key_chord);
    }, nb::arg("key_chord"));

mod_imgui.def("get_key_pressed_amount",
    [] (ImGuiKey key, float repeat_delay, float rate) -> int {
        return ImGui::GetKeyPressedAmount(key, repeat_delay, rate);
    }, nb::arg("key"), nb::arg("repeat_delay"), nb::arg("rate"));

mod_imgui.def("get_key_name",
    [] (ImGuiKey key) -> const char* {
        return ImGui::GetKeyName(key);
    }, nb::arg("key"));

mod_imgui.def("set_next_frame_want_capture_keyboard",
    [] (bool want_capture_keyboard) -> void {
        return ImGui::SetNextFrameWantCaptureKeyboard(want_capture_keyboard);
    }, nb::arg("want_capture_keyboard"));

mod_imgui.def("shortcut",
    [] (ImGuiKeyChord key_chord, ImGuiInputFlags flags) -> bool {
        return ImGui::Shortcut(key_chord, flags);
    }, nb::arg("key_chord"), nb::arg("flags") = 0);

mod_imgui.def("set_next_item_shortcut",
    [] (ImGuiKeyChord key_chord, ImGuiInputFlags flags) -> void {
        return ImGui::SetNextItemShortcut(key_chord, flags);
    }, nb::arg("key_chord"), nb::arg("flags") = 0);

mod_imgui.def("set_item_key_owner",
    [] (ImGuiKey key) -> void {
        return ImGui::SetItemKeyOwner(key);
    }, nb::arg("key"));

mod_imgui.def("is_mouse_down",
    [] (ImGuiMouseButton button) -> bool {
        return ImGui::IsMouseDown(button);
    }, nb::arg("button"));

mod_imgui.def("is_mouse_clicked",
    [] (ImGuiMouseButton button, bool repeat) -> bool {
        return ImGui::IsMouseClicked(button, repeat);
    }, nb::arg("button"), nb::arg("repeat") = false);

mod_imgui.def("is_mouse_released",
    [] (ImGuiMouseButton button) -> bool {
        return ImGui::IsMouseReleased(button);
    }, nb::arg("button"));

mod_imgui.def("is_mouse_double_clicked",
    [] (ImGuiMouseButton button) -> bool {
        return ImGui::IsMouseDoubleClicked(button);
    }, nb::arg("button"));

mod_imgui.def("is_mouse_released_with_delay",
    [] (ImGuiMouseButton button, float delay) -> bool {
        return ImGui::IsMouseReleasedWithDelay(button, delay);
    }, nb::arg("button"), nb::arg("delay"));

mod_imgui.def("get_mouse_clicked_count",
    [] (ImGuiMouseButton button) -> int {
        return ImGui::GetMouseClickedCount(button);
    }, nb::arg("button"));

mod_imgui.def("is_mouse_hovering_rect",
    [] (ImVec2 r_min, ImVec2 r_max, bool clip) -> bool {
        return ImGui::IsMouseHoveringRect(r_min, r_max, clip);
    }, nb::arg("r_min"), nb::arg("r_max"), nb::arg("clip") = true);

mod_imgui.def("is_any_mouse_down",
    [] () -> bool {
        return ImGui::IsAnyMouseDown();
    });

mod_imgui.def("get_mouse_pos",
    [] () -> ImVec2 {
        return ImGui::GetMousePos();
    });

mod_imgui.def("get_mouse_pos_on_opening_current_popup",
    [] () -> ImVec2 {
        return ImGui::GetMousePosOnOpeningCurrentPopup();
    });

mod_imgui.def("is_mouse_dragging",
    [] (ImGuiMouseButton button, float lock_threshold) -> bool {
        return ImGui::IsMouseDragging(button, lock_threshold);
    }, nb::arg("button"), nb::arg("lock_threshold") = -1.0);

mod_imgui.def("get_mouse_drag_delta",
    [] (ImGuiMouseButton button, float lock_threshold) -> ImVec2 {
        return ImGui::GetMouseDragDelta(button, lock_threshold);
    }, nb::arg("button") = 0, nb::arg("lock_threshold") = -1.0);

mod_imgui.def("reset_mouse_drag_delta",
    [] (ImGuiMouseButton button) -> void {
        return ImGui::ResetMouseDragDelta(button);
    }, nb::arg("button") = 0);

mod_imgui.def("get_mouse_cursor",
    [] () -> ImGuiMouseCursor {
        return ImGui::GetMouseCursor();
    });

mod_imgui.def("set_mouse_cursor",
    [] (ImGuiMouseCursor cursor_type) -> void {
        return ImGui::SetMouseCursor(cursor_type);
    }, nb::arg("cursor_type"));

mod_imgui.def("set_next_frame_want_capture_mouse",
    [] (bool want_capture_mouse) -> void {
        return ImGui::SetNextFrameWantCaptureMouse(want_capture_mouse);
    }, nb::arg("want_capture_mouse"));

mod_imgui.def("get_clipboard_text",
    [] () -> const char* {
        return ImGui::GetClipboardText();
    });

mod_imgui.def("set_clipboard_text",
    [] (const char* text) -> void {
        return ImGui::SetClipboardText(text);
    }, nb::arg("text"));

mod_imgui.def("load_ini_settings_from_disk",
    [] (const char* ini_filename) -> void {
        return ImGui::LoadIniSettingsFromDisk(ini_filename);
    }, nb::arg("ini_filename"));

mod_imgui.def("load_ini_settings_from_memory",
    [] (const char* ini_data, size_t ini_size) -> void {
        return ImGui::LoadIniSettingsFromMemory(ini_data, ini_size);
    }, nb::arg("ini_data"), nb::arg("ini_size") = 0);

mod_imgui.def("save_ini_settings_to_disk",
    [] (const char* ini_filename) -> void {
        return ImGui::SaveIniSettingsToDisk(ini_filename);
    }, nb::arg("ini_filename"));

mod_imgui.def("debug_text_encoding",
    [] (const char* text) -> void {
        return ImGui::DebugTextEncoding(text);
    }, nb::arg("text"));

mod_imgui.def("debug_flash_style_color",
    [] (ImGuiCol idx) -> void {
        return ImGui::DebugFlashStyleColor(idx);
    }, nb::arg("idx"));

mod_imgui.def("debug_start_item_picker",
    [] () -> void {
        return ImGui::DebugStartItemPicker();
    });

mod_imgui.def("debug_check_version_and_data_layout",
    [] (const char* version_str, size_t sz_io, size_t sz_style, size_t sz_vec2, size_t sz_vec4, size_t sz_drawvert, size_t sz_drawidx) -> bool {
        return ImGui::DebugCheckVersionAndDataLayout(version_str, sz_io, sz_style, sz_vec2, sz_vec4, sz_drawvert, sz_drawidx);
    }, nb::arg("version_str"), nb::arg("sz_io"), nb::arg("sz_style"), nb::arg("sz_vec2"), nb::arg("sz_vec4"), nb::arg("sz_drawvert"), nb::arg("sz_drawidx"));

mod_imgui.def("debug_log",
    [] (const char* fmt) -> void {
        return ImGui::DebugLog("%s", fmt);
    }, nb::arg("fmt"));

mod_imgui.def("find_viewport_by_id",
    [] (ImGuiID id) -> ImGuiViewport* {
        return ImGui::FindViewportByID(id);
    }, nb::arg("id"));

auto drawlist_class = nb::class_<DrawList>(mod_imgui, "DrawList",
    nb::intrusive_ptr<DrawList>([](DrawList *o, PyObject *po) noexcept { o->set_self_py(po); }))

    .def("push_clip_rect",
    [] (DrawList* self, ImVec2 clip_rect_min, ImVec2 clip_rect_max, bool intersect_with_current_clip_rect) -> void {
        return self->list->PushClipRect(clip_rect_min, clip_rect_max, intersect_with_current_clip_rect);
    }, nb::arg("clip_rect_min"), nb::arg("clip_rect_max"), nb::arg("intersect_with_current_clip_rect") = false)
    .def("push_clip_rect_full_screen",
    [] (DrawList* self) -> void {
        return self->list->PushClipRectFullScreen();
    })
    .def("pop_clip_rect",
    [] (DrawList* self) -> void {
        return self->list->PopClipRect();
    })
    .def("push_texture_id",
    [] (DrawList* self, ImTextureID texture_id) -> void {
        return self->list->PushTextureID(texture_id);
    }, nb::arg("texture_id"))
    .def("pop_texture_id",
    [] (DrawList* self) -> void {
        return self->list->PopTextureID();
    })
    .def("get_clip_rect_min",
    [] (DrawList* self) -> ImVec2 {
        return self->list->GetClipRectMin();
    })
    .def("get_clip_rect_max",
    [] (DrawList* self) -> ImVec2 {
        return self->list->GetClipRectMax();
    })
    .def("add_line",
    [] (DrawList* self, ImVec2 p1, ImVec2 p2, Color col, float thickness) -> void {
        return self->list->AddLine(p1, p2, col, thickness);
    }, nb::arg("p1"), nb::arg("p2"), nb::arg("col"), nb::arg("thickness") = 1.0)
    .def("add_rect",
    [] (DrawList* self, ImVec2 p_min, ImVec2 p_max, Color col, float rounding, ImDrawFlags flags, float thickness) -> void {
        return self->list->AddRect(p_min, p_max, col, rounding, flags, thickness);
    }, nb::arg("p_min"), nb::arg("p_max"), nb::arg("col"), nb::arg("rounding") = 0.0, nb::arg("flags") = 0, nb::arg("thickness") = 1.0)
    .def("add_rect_filled",
    [] (DrawList* self, ImVec2 p_min, ImVec2 p_max, Color col, float rounding, ImDrawFlags flags) -> void {
        return self->list->AddRectFilled(p_min, p_max, col, rounding, flags);
    }, nb::arg("p_min"), nb::arg("p_max"), nb::arg("col"), nb::arg("rounding") = 0.0, nb::arg("flags") = 0)
    .def("add_rect_filled_multi_color",
    [] (DrawList* self, ImVec2 p_min, ImVec2 p_max, Color col_upr_left, Color col_upr_right, Color col_bot_right, Color col_bot_left) -> void {
        return self->list->AddRectFilledMultiColor(p_min, p_max, col_upr_left, col_upr_right, col_bot_right, col_bot_left);
    }, nb::arg("p_min"), nb::arg("p_max"), nb::arg("col_upr_left"), nb::arg("col_upr_right"), nb::arg("col_bot_right"), nb::arg("col_bot_left"))
    .def("add_quad",
    [] (DrawList* self, ImVec2 p1, ImVec2 p2, ImVec2 p3, ImVec2 p4, Color col, float thickness) -> void {
        return self->list->AddQuad(p1, p2, p3, p4, col, thickness);
    }, nb::arg("p1"), nb::arg("p2"), nb::arg("p3"), nb::arg("p4"), nb::arg("col"), nb::arg("thickness") = 1.0)
    .def("add_quad_filled",
    [] (DrawList* self, ImVec2 p1, ImVec2 p2, ImVec2 p3, ImVec2 p4, Color col) -> void {
        return self->list->AddQuadFilled(p1, p2, p3, p4, col);
    }, nb::arg("p1"), nb::arg("p2"), nb::arg("p3"), nb::arg("p4"), nb::arg("col"))
    .def("add_triangle",
    [] (DrawList* self, ImVec2 p1, ImVec2 p2, ImVec2 p3, Color col, float thickness) -> void {
        return self->list->AddTriangle(p1, p2, p3, col, thickness);
    }, nb::arg("p1"), nb::arg("p2"), nb::arg("p3"), nb::arg("col"), nb::arg("thickness") = 1.0)
    .def("add_triangle_filled",
    [] (DrawList* self, ImVec2 p1, ImVec2 p2, ImVec2 p3, Color col) -> void {
        return self->list->AddTriangleFilled(p1, p2, p3, col);
    }, nb::arg("p1"), nb::arg("p2"), nb::arg("p3"), nb::arg("col"))
    .def("add_circle",
    [] (DrawList* self, ImVec2 center, float radius, Color col, int num_segments, float thickness) -> void {
        return self->list->AddCircle(center, radius, col, num_segments, thickness);
    }, nb::arg("center"), nb::arg("radius"), nb::arg("col"), nb::arg("num_segments") = 0, nb::arg("thickness") = 1.0)
    .def("add_circle_filled",
    [] (DrawList* self, ImVec2 center, float radius, Color col, int num_segments) -> void {
        return self->list->AddCircleFilled(center, radius, col, num_segments);
    }, nb::arg("center"), nb::arg("radius"), nb::arg("col"), nb::arg("num_segments") = 0)
    .def("add_ngon",
    [] (DrawList* self, ImVec2 center, float radius, Color col, int num_segments, float thickness) -> void {
        return self->list->AddNgon(center, radius, col, num_segments, thickness);
    }, nb::arg("center"), nb::arg("radius"), nb::arg("col"), nb::arg("num_segments"), nb::arg("thickness") = 1.0)
    .def("add_ngon_filled",
    [] (DrawList* self, ImVec2 center, float radius, Color col, int num_segments) -> void {
        return self->list->AddNgonFilled(center, radius, col, num_segments);
    }, nb::arg("center"), nb::arg("radius"), nb::arg("col"), nb::arg("num_segments"))
    .def("add_ellipse",
    [] (DrawList* self, ImVec2 center, ImVec2 radius, Color col, float rot, int num_segments, float thickness) -> void {
        return self->list->AddEllipse(center, radius, col, rot, num_segments, thickness);
    }, nb::arg("center"), nb::arg("radius"), nb::arg("col"), nb::arg("rot") = 0.0, nb::arg("num_segments") = 0, nb::arg("thickness") = 1.0)
    .def("add_ellipse_filled",
    [] (DrawList* self, ImVec2 center, ImVec2 radius, Color col, float rot, int num_segments) -> void {
        return self->list->AddEllipseFilled(center, radius, col, rot, num_segments);
    }, nb::arg("center"), nb::arg("radius"), nb::arg("col"), nb::arg("rot") = 0.0, nb::arg("num_segments") = 0)
    .def("add_text",
    [] (DrawList* self, ImVec2 pos, Color col, const char* text) -> void {
        return self->list->AddText(pos, col, text, nullptr);
    }, nb::arg("pos"), nb::arg("col"), nb::arg("text"))
    .def("add_bezier_cubic",
    [] (DrawList* self, ImVec2 p1, ImVec2 p2, ImVec2 p3, ImVec2 p4, Color col, float thickness, int num_segments) -> void {
        return self->list->AddBezierCubic(p1, p2, p3, p4, col, thickness, num_segments);
    }, nb::arg("p1"), nb::arg("p2"), nb::arg("p3"), nb::arg("p4"), nb::arg("col"), nb::arg("thickness"), nb::arg("num_segments") = 0)
    .def("add_bezier_quadratic",
    [] (DrawList* self, ImVec2 p1, ImVec2 p2, ImVec2 p3, Color col, float thickness, int num_segments) -> void {
        return self->list->AddBezierQuadratic(p1, p2, p3, col, thickness, num_segments);
    }, nb::arg("p1"), nb::arg("p2"), nb::arg("p3"), nb::arg("col"), nb::arg("thickness"), nb::arg("num_segments") = 0)
    .def("add_polyline",
    [] (DrawList* self, const ImVec2* points, int num_points, Color col, ImDrawFlags flags, float thickness) -> void {
        return self->list->AddPolyline(points, num_points, col, flags, thickness);
    }, nb::arg("points"), nb::arg("num_points"), nb::arg("col"), nb::arg("flags"), nb::arg("thickness"))
    .def("add_convex_poly_filled",
    [] (DrawList* self, const ImVec2* points, int num_points, Color col) -> void {
        return self->list->AddConvexPolyFilled(points, num_points, col);
    }, nb::arg("points"), nb::arg("num_points"), nb::arg("col"))
    .def("add_concave_poly_filled",
    [] (DrawList* self, const ImVec2* points, int num_points, Color col) -> void {
        return self->list->AddConcavePolyFilled(points, num_points, col);
    }, nb::arg("points"), nb::arg("num_points"), nb::arg("col"))
    .def("add_image",
    [] (DrawList* self, ImTextureID user_texture_id, ImVec2 p_min, ImVec2 p_max, ImVec2 uv_min, ImVec2 uv_max, Color col) -> void {
        return self->list->AddImage(user_texture_id, p_min, p_max, uv_min, uv_max, col);
    }, nb::arg("user_texture_id"), nb::arg("p_min"), nb::arg("p_max"), nb::arg("uv_min") = nb::make_tuple(0, 0), nb::arg("uv_max") = nb::make_tuple(1, 1), nb::arg("col") = IM_COL32_WHITE)
    .def("add_image_quad",
    [] (DrawList* self, ImTextureID user_texture_id, ImVec2 p1, ImVec2 p2, ImVec2 p3, ImVec2 p4, ImVec2 uv1, ImVec2 uv2, ImVec2 uv3, ImVec2 uv4, Color col) -> void {
        return self->list->AddImageQuad(user_texture_id, p1, p2, p3, p4, uv1, uv2, uv3, uv4, col);
    }, nb::arg("user_texture_id"), nb::arg("p1"), nb::arg("p2"), nb::arg("p3"), nb::arg("p4"), nb::arg("uv1") = nb::make_tuple(0, 0), nb::arg("uv2") = nb::make_tuple(1, 0), nb::arg("uv3") = nb::make_tuple(1, 1), nb::arg("uv4") = nb::make_tuple(0, 1), nb::arg("col") = IM_COL32_WHITE)
    .def("add_image_rounded",
    [] (DrawList* self, ImTextureID user_texture_id, ImVec2 p_min, ImVec2 p_max, ImVec2 uv_min, ImVec2 uv_max, Color col, float rounding, ImDrawFlags flags) -> void {
        return self->list->AddImageRounded(user_texture_id, p_min, p_max, uv_min, uv_max, col, rounding, flags);
    }, nb::arg("user_texture_id"), nb::arg("p_min"), nb::arg("p_max"), nb::arg("uv_min"), nb::arg("uv_max"), nb::arg("col"), nb::arg("rounding"), nb::arg("flags") = 0)
    .def("path_clear",
    [] (DrawList* self) -> void {
        return self->list->PathClear();
    })
    .def("path_line_to",
    [] (DrawList* self, ImVec2 pos) -> void {
        return self->list->PathLineTo(pos);
    }, nb::arg("pos"))
    .def("path_line_to_merge_duplicate",
    [] (DrawList* self, ImVec2 pos) -> void {
        return self->list->PathLineToMergeDuplicate(pos);
    }, nb::arg("pos"))
    .def("path_fill_convex",
    [] (DrawList* self, Color col) -> void {
        return self->list->PathFillConvex(col);
    }, nb::arg("col"))
    .def("path_fill_concave",
    [] (DrawList* self, Color col) -> void {
        return self->list->PathFillConcave(col);
    }, nb::arg("col"))
    .def("path_stroke",
    [] (DrawList* self, Color col, ImDrawFlags flags, float thickness) -> void {
        return self->list->PathStroke(col, flags, thickness);
    }, nb::arg("col"), nb::arg("flags") = 0, nb::arg("thickness") = 1.0)
    .def("path_arc_to",
    [] (DrawList* self, ImVec2 center, float radius, float a_min, float a_max, int num_segments) -> void {
        return self->list->PathArcTo(center, radius, a_min, a_max, num_segments);
    }, nb::arg("center"), nb::arg("radius"), nb::arg("a_min"), nb::arg("a_max"), nb::arg("num_segments") = 0)
    .def("path_arc_to_fast",
    [] (DrawList* self, ImVec2 center, float radius, int a_min_of_12, int a_max_of_12) -> void {
        return self->list->PathArcToFast(center, radius, a_min_of_12, a_max_of_12);
    }, nb::arg("center"), nb::arg("radius"), nb::arg("a_min_of_12"), nb::arg("a_max_of_12"))
    .def("path_elliptical_arc_to",
    [] (DrawList* self, ImVec2 center, ImVec2 radius, float rot, float a_min, float a_max, int num_segments) -> void {
        return self->list->PathEllipticalArcTo(center, radius, rot, a_min, a_max, num_segments);
    }, nb::arg("center"), nb::arg("radius"), nb::arg("rot"), nb::arg("a_min"), nb::arg("a_max"), nb::arg("num_segments") = 0)
    .def("path_bezier_cubic_curve_to",
    [] (DrawList* self, ImVec2 p2, ImVec2 p3, ImVec2 p4, int num_segments) -> void {
        return self->list->PathBezierCubicCurveTo(p2, p3, p4, num_segments);
    }, nb::arg("p2"), nb::arg("p3"), nb::arg("p4"), nb::arg("num_segments") = 0)
    .def("path_bezier_quadratic_curve_to",
    [] (DrawList* self, ImVec2 p2, ImVec2 p3, int num_segments) -> void {
        return self->list->PathBezierQuadraticCurveTo(p2, p3, num_segments);
    }, nb::arg("p2"), nb::arg("p3"), nb::arg("num_segments") = 0)
    .def("path_rect",
    [] (DrawList* self, ImVec2 rect_min, ImVec2 rect_max, float rounding, ImDrawFlags flags) -> void {
        return self->list->PathRect(rect_min, rect_max, rounding, flags);
    }, nb::arg("rect_min"), nb::arg("rect_max"), nb::arg("rounding") = 0.0, nb::arg("flags") = 0)
    .def("add_draw_cmd",
    [] (DrawList* self) -> void {
        return self->list->AddDrawCmd();
    })
    .def("clone_output",
    [] (DrawList* self) -> DrawList* {
        return new DrawList(self->list->CloneOutput());
    })
    .def("channels_split",
    [] (DrawList* self, int count) -> void {
        return self->list->ChannelsSplit(count);
    }, nb::arg("count"))
    .def("channels_merge",
    [] (DrawList* self) -> void {
        return self->list->ChannelsMerge();
    })
    .def("channels_set_current",
    [] (DrawList* self, int n) -> void {
        return self->list->ChannelsSetCurrent(n);
    }, nb::arg("n"))
    .def("prim_reserve",
    [] (DrawList* self, int idx_count, int vtx_count) -> void {
        return self->list->PrimReserve(idx_count, vtx_count);
    }, nb::arg("idx_count"), nb::arg("vtx_count"))
    .def("prim_unreserve",
    [] (DrawList* self, int idx_count, int vtx_count) -> void {
        return self->list->PrimUnreserve(idx_count, vtx_count);
    }, nb::arg("idx_count"), nb::arg("vtx_count"))
    .def("prim_rect",
    [] (DrawList* self, ImVec2 a, ImVec2 b, Color col) -> void {
        return self->list->PrimRect(a, b, col);
    }, nb::arg("a"), nb::arg("b"), nb::arg("col"))
    .def("prim_write_vtx",
    [] (DrawList* self, ImVec2 pos, ImVec2 uv, Color col) -> void {
        return self->list->PrimWriteVtx(pos, uv, col);
    }, nb::arg("pos"), nb::arg("uv"), nb::arg("col"))
    .def("prim_write_idx",
    [] (DrawList* self, ImDrawIdx idx) -> void {
        return self->list->PrimWriteIdx(idx);
    }, nb::arg("idx"))
    .def("prim_vtx",
    [] (DrawList* self, ImVec2 pos, ImVec2 uv, Color col) -> void {
        return self->list->PrimVtx(pos, uv, col);
    }, nb::arg("pos"), nb::arg("uv"), nb::arg("col"))
;
mod_imgui.def("image_im_vec4",
    [] (ImTextureID user_texture_id, ImVec2 image_size, ImVec2 uv0, ImVec2 uv1, ImVec4 tint_col, ImVec4 border_col) -> void {
        return ImGui::Image(user_texture_id, image_size, uv0, uv1, tint_col, border_col);
    }, nb::arg("user_texture_id"), nb::arg("image_size"), nb::arg("uv0"), nb::arg("uv1"), nb::arg("tint_col"), nb::arg("border_col"));

mod_imgui.def("push_button_repeat",
    [] (bool repeat) -> void {
        return ImGui::PushButtonRepeat(repeat);
    }, nb::arg("repeat"));

mod_imgui.def("pop_button_repeat",
    [] () -> void {
        return ImGui::PopButtonRepeat();
    });

mod_imgui.def("push_tab_stop",
    [] (bool tab_stop) -> void {
        return ImGui::PushTabStop(tab_stop);
    }, nb::arg("tab_stop"));

mod_imgui.def("pop_tab_stop",
    [] () -> void {
        return ImGui::PopTabStop();
    });

mod_imgui.def("get_content_region_max",
    [] () -> ImVec2 {
        return ImGui::GetContentRegionMax();
    });

mod_imgui.def("get_window_content_region_min",
    [] () -> ImVec2 {
        return ImGui::GetWindowContentRegionMin();
    });

mod_imgui.def("get_window_content_region_max",
    [] () -> ImVec2 {
        return ImGui::GetWindowContentRegionMax();
    });

mod_imgui.def("begin_child_frame",
    [] (ImGuiID id, ImVec2 size, ImGuiWindowFlags window_flags) -> bool {
        return ImGui::BeginChildFrame(id, size, window_flags);
    }, nb::arg("id"), nb::arg("size"), nb::arg("window_flags") = 0);

mod_imgui.def("end_child_frame",
    [] () -> void {
        return ImGui::EndChildFrame();
    });

mod_imgui.def("show_stack_tool_window",
    [] (std::optional<bool> p_open) -> std::optional<bool> {
        ImGui::ShowStackToolWindow(p_open.has_value() ? &p_open.value() : NULL);
        return p_open;
    }, nb::arg("p_open") = nb::none());

mod_imgui.def("set_item_allow_overlap",
    [] () -> void {
        return ImGui::SetItemAllowOverlap();
    });

mod_imgui.def("push_allow_keyboard_focus",
    [] (bool tab_stop) -> void {
        return ImGui::PushAllowKeyboardFocus(tab_stop);
    }, nb::arg("tab_stop"));

mod_imgui.def("pop_allow_keyboard_focus",
    [] () -> void {
        return ImGui::PopAllowKeyboardFocus();
    });

