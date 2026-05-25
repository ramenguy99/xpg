# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

from pyxpg import Key, Modifiers, MouseButton


@dataclass(frozen=True)
class KeyBinding:
    key: Key
    mods: Modifiers = Modifiers.NONE

    def is_active(self, key: Key, mods: Modifiers) -> bool:
        return self.key == key and self.mods == mods


@dataclass(frozen=True)
class MouseButtonBinding:
    button: MouseButton
    mods: Modifiers = Modifiers.NONE

    def is_active(self, button: MouseButton, mods: Modifiers) -> bool:
        return self.button == button and self.mods == mods


@dataclass
class KeyMap:
    exit: KeyBinding = KeyBinding(Key.ESCAPE)

    # Playback
    toggle_play_pause: KeyBinding = KeyBinding(Key.SPACE)
    previous_frame: KeyBinding = KeyBinding(Key.COMMA)
    next_frame: KeyBinding = KeyBinding(Key.PERIOD)
    first_frame: KeyBinding = KeyBinding(Key.COMMA, Modifiers.SHIFT)
    last_frame: KeyBinding = KeyBinding(Key.PERIOD, Modifiers.SHIFT)

    # Renderer
    toggle_path_tracer: KeyBinding = KeyBinding(Key.R, Modifiers.CTRL)

    # Camera control
    camera_rotate: MouseButtonBinding = MouseButtonBinding(MouseButton.LEFT)
    camera_pan: MouseButtonBinding = MouseButtonBinding(MouseButton.RIGHT)
    camera_zoom_modifiers: Modifiers = Modifiers.NONE
    camera_zoom_move_modifiers: Modifiers = Modifiers.SHIFT

    ortho_view_positive_x: KeyBinding = KeyBinding(Key.F1, Modifiers.NONE)
    ortho_view_positive_y: KeyBinding = KeyBinding(Key.F2, Modifiers.NONE)
    ortho_view_positive_z: KeyBinding = KeyBinding(Key.F3, Modifiers.NONE)
    ortho_view_negative_x: KeyBinding = KeyBinding(Key.F1, Modifiers.SHIFT)
    ortho_view_negative_y: KeyBinding = KeyBinding(Key.F2, Modifiers.SHIFT)
    ortho_view_negative_z: KeyBinding = KeyBinding(Key.F3, Modifiers.SHIFT)
