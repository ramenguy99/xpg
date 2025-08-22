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
    next_frame: KeyBinding = KeyBinding(Key.PERIOD)
    previous_frame: KeyBinding = KeyBinding(Key.COMMA)

    # Camera control
    camera_rotate: MouseButtonBinding = MouseButtonBinding(MouseButton.LEFT)
    camera_pan: MouseButtonBinding = MouseButtonBinding(MouseButton.RIGHT)
    camera_zoom_modifiers: Modifiers = Modifiers.NONE
    camera_zoom_move_modifiers: Modifiers = Modifiers.SHIFT
