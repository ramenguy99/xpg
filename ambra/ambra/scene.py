from dataclasses import dataclass

from enum import Enum, auto
from typing import Optional, TypeVar, Generic, Union, List, Tuple
import numpy as np
import sys
from pyglm.glm import vec3, vec2, quat

T = TypeVar('T')

if sys.version_info >= (3, 12):
    from collections.abc import Buffer
    PropertyData = Union[List, Tuple, np.ndarray, Buffer]
else:
    PropertyData = Union[List, Tuple, np.ndarray]


# class AnimationInterpolation(Enum):
#     NEAREST = auto()

#     # For linear quantities
#     LINEAR = auto()
#     CUBIC = auto()

#     # For rotations
#     NLERP = auto()
#     SLERP = auto()


class AnimationBoundary(Enum):
    HOLD = auto()
    REPEAT = auto()
    MIRROR = auto()

    def map(self, frame: int, n: int) -> int:
        if self == AnimationBoundary.HOLD:
            return min(frame, n - 1)
        elif self == AnimationBoundary.MIRROR:
            return (frame + n - 1) % 2 * n - 1
        elif self == AnimationBoundary.REPEAT:
            return frame % n
        else:
            raise ValueError(f"Unhandled enum variant: {self.boundary}")

@dataclass
class Animation:
    # interpolation = AnimationInterpolation.NEAREST
    boundary: AnimationBoundary

    def get_frame_index(self, n: int, global_time: float, global_frame: int) -> int:
        return 0

    def max_animation_time(self, n: int, fps: float) -> float:
        return 0

# @dataclass
# class SampledAnimation(Animation):
#     samples: np.typing.ArrayLike[float] # list of animation samples in seconds

#     def get_frame_index(self, time: float) -> int:
#         return 



@dataclass
class ConstantSpeedAnimation(Animation):
    frames_per_second: float # frame time in seconds

    def get_frame_index(self, n: int, global_time: float, global_frame: int) -> int:
        if n == 0: return -1
        return self.boundary.map(int(global_time * self.frames_per_second), n)

    def max_animation_time(self, n: int, fps: float) -> float:
        return n / self.frames_per_second

@dataclass
class FrameAnimation(Animation):
    def get_frame_index(self, n:int, global_time: float, global_frame: int) -> int:
        if n == 0: return -1
        return self.boundary.map(global_frame, n)

    def max_animation_time(self, n: int, frames_per_second: float) -> float:
        return n / frames_per_second

class Property(Generic[T]):
    def __init__(self, num_frames: int, animation: Optional[Animation] = None, name: str = "" , prefer_preupload: Optional[bool] = None):
        self.num_frames = num_frames
        self.name = name
        self.animation = animation or Animation(AnimationBoundary.HOLD)
        self.prefer_preupload = prefer_preupload

        self.current_time = 0
        self.current_frame = 0
    
    def get_frame_index(self, time: float, playback_frame: int) -> int:
        return self.animation.get_frame_index(self.num_frames, time, playback_frame)

    def get_frame(self, time: float, frame: int) -> T:
        return None

    def get_frame_by_index(self, frame: int) -> T:
        return None
    
    def max_animation_time(self, fps: float) -> float:
        return self.animation.max_animation_time(self.num_frames, fps)
    
    def update(self, time: float, frame: int) -> T:
        self.current_time = time
        self.current_frame = frame
    
    def get_current(self) -> T:
        return self.get_frame(self.current_time, self.current_frame)
    

class DataProperty(Property):
    def __init__(self, data: PropertyData, animation: Optional[Animation] = None, name: str = "", prefer_preupload: Optional[bool] = None):
        super().__init__(len(data), animation, name, prefer_preupload)
        self.data = data
    
    def get_frame(self, time: float, frame: int) -> T:
        return self.data[self.get_frame_index(time, frame)]

    def get_frame_by_index(self, frame: int) -> T:
        return self.data[frame]


class StreamingProperty(Property):
    def __init__(self, num_frames: int, animation: Optional[Animation] = None, name: str = "", prefer_preupload: Optional[bool] = None):
        super().__init__(num_frames, animation, name, prefer_preupload)

    def get_frame(self, time: float, frame: int) -> T:
        raise NotImplemented()
        # return self.load(self.get_frame_index(time, frame))
    
    def get_frame_by_index(self, frame: int) -> T:
        return NotImplemented()

def shape_match(expected_shape: Tuple[int], got_shape: Tuple[int]) -> bool:
    if len(expected_shape) != len(got_shape):
        return False
    for a, b in zip(expected_shape, got_shape):
        if a != -1 and a != b:
            return False
    return True

def as_property(value: Union[Property[T], PropertyData], dtype: Optional[np.dtype] = None, shape: Optional[Tuple[int]] = None, animation: Optional[Animation] = None, name: str = ""):
    if isinstance(value, Property):
        # TODO: typecheck arrays if data property here?
        # TODO: override name and anim if passed?
        return value
    else:
        if shape is None:
            value = np.asarray([value], dtype, order="C")
            if len(value.shape) > 1:
                raise ShapeException(value.shape[1:], (1,))
        else:
            if isinstance(value, Tuple) or isinstance(value, List):
                for i in range(len(value)):
                    a = np.asarray(value[i], dtype)
                    if shape_match(shape, a.shape):
                        raise ShapeException(shape, a.shape)
                    value[i] = a
            else:
                value = np.asarray(value, dtype, order="C")
                if shape_match(shape, value.shape):
                    # Implicitly add animation dimension
                    value = value[np.newaxis]
                elif shape_match(shape, value.shape[1:]):
                    # Shape already matches
                    pass
                else:
                    raise ShapeException(shape, value.shape)
        return DataProperty(value, animation, name)



"""
class Property(Property):
    def __init__(self, data: PropertyData[T], animation: Optional[Animation] = None, name: str = ""):
        self.data = data
        self.animation = animation or Animation(AnimationInterpolation.FLOOR, AnimationBoundary.HOLD)
        self.name = name
    
    def __getitem__(self, key):
        return self.data.__getitem__(key)
    
    def __len__(self):
        return self.data.__len__()
    
    def sample(self, time: float) -> T:
        # NOTE: this is for CPU usage only, when rendering interpolation is done in an object specific
        # way (e.g. in shaders), therefor most objects will not support all animation types for all properties.
        #
        # We put the burden of supporting something similar to this to the implementations, based on the same
        # animation 
        #
        # Technically we could think of a "high quality" rendering mode that uses this to support all interpolation
        # types on the CPU and does per-frame upload.
        pass
    
    def max_count(self):
        if isinstance(self.data, Tuple) or isinstance(self.data, List):
            return max([len(self.data[i]) for i in range(len(self.data))])
        else:
            return self.shape[1]
    
    def max_size(self):
        return (self.data[0].dtype.itemsize if len(self.data) else 0) * self.max_count()
"""


class ShapeException(Exception):
    def __init__(self, expected: Tuple[int], got: Tuple[int]):
        self.expected = expected
        self.got = got
    
    def __str__(self):
        return f"Shape mismatch. Expected: {self.expected}, got: {self.got}"


class Object:
    def __init__(self, name: str):
        self.name = name
        self.children: List["Object"] = []
        self.properties: List[Property] = []
    
    def add_property(self, prop: Property, dtype: Optional[np.dtype] = None, shape: Optional[Tuple[int]] = None, animation: Optional[Animation] = None, name: str = "") -> Property:
        property = as_property(prop, dtype, shape, animation, name)
        self.properties.append(property)
        return property
    
    # def create(self, renderer: Renderer):
    def create(self, renderer):
        pass

    def update(self, time: float, frame: int):
        pass

    # def render(self, renderer: Renderer):
    def render(self, renderer, frame):
        pass

    # def destroy(self, renderer: Renderer):
    def destroy(self):
        pass
    
    def _set_time(self, time):
        for p in self.properties:
            pass


class Object2D(Object):
    def __init__(self,
                 name: str,
                 translation: Optional[Property[vec2]] = None,
                 rotation: Optional[Property[float]] = None,
                 scale: Optional[Property[vec2]] = None):
        super().__init__(name)
        self.translation = as_property(translation) if translation else Property(np.array([[0, 0]], np.float32))
        self.rotation = as_property(rotation) if rotation else Property(np.array([[1]], np.float32))
        self.scale = as_property(scale) if scale else Property(np.array([[1, 1]], np.float32))

        self.add_property(self.translation)
        self.add_property(self.rotation)
        self.add_property(self.scale)


class Object3D(Object):
    def __init__(self,
                 name: str,
                 translation: Optional[Property[vec3]] = None,
                 rotation: Optional[Property[quat]] = None,
                 scale: Optional[Property[vec3]] = None):
        super().__init__(name)
        self.translation = as_property(translation) if translation else Property(np.array([[0, 0, 0]], np.float32))
        self.rotation = as_property(rotation) if rotation else Property(np.array([[1, 0, 0, 0]], np.float32))
        self.scale = as_property(scale) if scale else Property(np.array([[1, 1, 1]], np.float32))

        self.add_property(self.translation)
        self.add_property(self.rotation)
        self.add_property(self.scale)


class Scene:
    def __init__(self, name):
        self.name = name
        self.objects = []
    
    # TODO: visitor helpers?

    def max_animation_time(self) -> float:
        time = 0
        def max_animation_time_recursive(o: Object):
            global time
            time = max(time, max(p.max_animation_time() for p in o.properties))

            for c in o.children:
                max_animation_time_recursive(c)

        for o in self.objects:
            max_animation_time_recursive(o)

        return time
    
    def update(self, time: float, frame: int):
        def update_recursive(parent: Object):
            parent.update(time, frame)
            for c in parent.children:
                update_recursive(c)

        for o in self.objects:
            update_recursive(o)