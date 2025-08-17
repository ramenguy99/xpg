from pyxpg import Key, Modifiers, MouseButton
from dataclasses import dataclass

@dataclass(frozen=True)
class KeyBinding:
    key: Key
    mods: Modifiers = Modifiers.NONE

    def is_active(self, key: Key, mods: Modifiers) -> bool:
        return self.key == key and self.mods == mods

@dataclass(frozen=True)
class MouseButtonBinding:
    key: MouseButton
    mods: Modifiers = Modifiers.NONE

@dataclass
class KeyMap:
    exit: KeyBinding = KeyBinding(Key.ESCAPE)

    # Playback
    toggle_play_pause: KeyBinding = KeyBinding(Key.SPACE)
    next_frame: KeyBinding = KeyBinding(Key.PERIOD)
    previous_frame: KeyBinding = KeyBinding(Key.COMMA)


    # Camera control
    camera_rotate: MouseButtonBinding = MouseButtonBinding(MouseButton.RIGHT)
    camera_pan: MouseButtonBinding = MouseButtonBinding(MouseButton.RIGHT, mods=Modifiers.SHIFT)

