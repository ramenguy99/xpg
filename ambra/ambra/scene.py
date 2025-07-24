from dataclasses import dataclass

from enum import Enum, auto
from typing import Optional, TypeVar, Generic, Union, List, Tuple, Callable
import numpy as np
import sys
from pyglm.glm import vec3, vec2, quat
from .transform3d import Transform

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
            dr = frame % (2 * n)
            if dr >= n:
                return 2 * n - dr - 1
            else:
                return dr
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
        self.animation = animation or FrameAnimation(AnimationBoundary.HOLD)
        self.prefer_preupload = prefer_preupload

        self.current_time = 0
        self.current_frame = 0

    def update(self, time: float, frame: int) -> T:
        self.current_time = time
        self.current_frame = self.get_frame_index(time, frame)

    def max_animation_time(self, fps: float) -> float:
        a = self.animation.max_animation_time(self.num_frames, fps)
        return a

    def get_frame_index(self, time: float, playback_frame: int) -> int:
        return self.animation.get_frame_index(self.num_frames, time, playback_frame)

    def get_current(self) -> T:
        return self.get_frame(self.current_time, self.current_frame)

    def get_frame(self, time: float, frame: int) -> T:
        return None

    def get_frame_by_index(self, frame: int) -> T:
        return None


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
            value = np.atleast_1d(np.asarray(value, dtype, order="C"))
            if len(value.shape) > 1:
                raise ShapeException((1,), value.shape[1:])
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


class ShapeException(Exception):
    def __init__(self, expected: Tuple[int], got: Tuple[int]):
        self.expected = expected
        self.got = got

    def __str__(self):
        return f"Shape mismatch. Expected: {self.expected}, got: {self.got}"


_counter = 0

class Object:
    def __init__(self, name: Optional[str] = None):
        self.name = name or f"{type(self).__name__} - {Object.next_id()}"
        self.children: List["Object"] = []
        self.properties: List[Property] = []

        self.update_callbacks: List[Callable[[float, int], None]] = []
        self.destroy_callbacks: List[Callable[[None], None]] = []

    @staticmethod
    def next_id():
        global _counter
        _counter += 1
        return _counter

    def add_property(self, prop: Property, dtype: Optional[np.dtype] = None, shape: Optional[Tuple[int]] = None, animation: Optional[Animation] = None, name: str = "") -> Property:
        property = as_property(prop, dtype, shape, animation, name)
        self.properties.append(property)
        self.update_callbacks.append(lambda time, frame: property.update(time,frame))
        return property

    def create(self, renderer):
        pass

    def update(self, time: float, frame: int):
        for c in self.update_callbacks:
            c(time, frame)

    def update_transform(self, parent: "Object"):
        # TODO: can merge with udpate? Where to find parent? With link?
        pass

    def render(self, renderer, frame):
        pass

    def destroy(self):
        for c in self.destroy_callbacks:
            c()


class Object2D(Object):
    def __init__(self,
                 name: str,
                 translation: Optional[Property[vec2]] = None,
                 rotation: Optional[Property[float]] = None,
                 scale: Optional[Property[vec2]] = None):
        super().__init__(name)
        self.translation = as_property(translation or np.array([0, 0]), np.float32, (2,), name=f"translation")
        self.rotation = as_property(rotation or np.array([0]), np.float32, name=f"rotation")
        self.scale = as_property(scale or np.array([1, 1]), np.float32, (2,), name=f"scale")

        self.add_property(self.translation)
        self.add_property(self.rotation)
        self.add_property(self.scale)


class Object3D(Object):
    def __init__(self,
                 name: str,
                 translation: Optional[Property[np.ndarray]] = None,
                 rotation: Optional[Property[np.ndarray]] = None,
                 scale: Optional[Property[np.ndarray]] = None):
        super().__init__(name)
        self.translation = as_property(translation if translation is not None else np.array([0, 0, 0]), np.float32, (3,), name=f"translation")
        self.rotation = as_property(rotation if rotation is not None else np.array([1, 0, 0, 0]), np.float32, (4,), name=f"rotation")
        self.scale = as_property(scale if scale is not None else np.array([1, 1, 1]), np.float32, (3,), name=f"scale")
        self.current_relative_transform = Transform(
            vec3(self.translation.get_current()),
            quat(self.rotation.get_current()),
            vec3(self.scale.get_current())
        )
        self.current_transform_mat4 = self.current_relative_transform.as_mat4()

        self.add_property(self.translation)
        self.add_property(self.rotation)
        self.add_property(self.scale)

    def update_transform(self, parent: "Object3D"):
        self.current_relative_transform = Transform(
            vec3(self.translation.get_current()),
            quat(self.rotation.get_current()),
            vec3(self.scale.get_current())
        )
        self.current_transform_mat4 = (parent.current_transform_mat4 if parent else np.eye(4)) @ self.current_relative_transform.as_mat4()

class Scene:
    def __init__(self, name):
        self.name = name
        self.objects = []

    def visit_objects(self, function: Callable[[Object], None]):
        def visit_recursive(p: Object, o: Object):
            function(p, o)
            for c in o.children:
                visit_recursive(o, c)

        for o in self.objects:
            visit_recursive(None, o)

    def max_animation_time(self, frames_per_second) -> float:
        time = 0
        def visit(p: Object, o: Object):
            nonlocal time
            time = max(time, max(p.max_animation_time(frames_per_second) for p in o.properties))
        self.visit_objects(visit)
        return time

    def update(self, time: float, frame: int):
        def visit(p: Object, o: Object):
            o.update(time, frame)
            o.update_transform(p)
        self.visit_objects(visit)