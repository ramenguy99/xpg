from dataclasses import dataclass

from enum import Enum, auto
from typing import Optional, TypeVar, Generic, Union, List, Tuple, Callable
import numpy as np
import sys
from pyglm.glm import vec3, vec2, quat
from .transform2d import Transform2D
from .transform3d import Transform3D

T = TypeVar('T')

if sys.version_info >= (3, 12):
    from collections.abc import Buffer
    PropertyData = Union[List[np.ndarray], np.ndarray, Buffer]
else:
    PropertyData = Union[List, np.ndarray]


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

    def get_frame_index(self, n: int, time: float, frame_index: int) -> int:
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

    def get_frame_index(self, n: int, time: float, frame_index: int) -> int:
        if n == 0: return -1
        return self.boundary.map(int(time * self.frames_per_second), n)

    def max_animation_time(self, n: int, fps: float) -> float:
        return n / self.frames_per_second


@dataclass
class FrameAnimation(Animation):
    def get_frame_index(self, n:int, time: float, frame_index: int) -> int:
        if n == 0: return -1
        return self.boundary.map(frame_index, n)

    def max_animation_time(self, n: int, frames_per_second: float) -> float:
        return n / frames_per_second


@dataclass
class UploadSettings:
    preupload: bool
    async_load: bool = False
    cpu_prefetch_count: int = 0
    gpu_prefetch_count: int = 0

class Property(Generic[T]):
    def __init__(self,
                 num_frames: int,
                 dtype: Optional[np.dtype] = None,
                 shape: Optional[Tuple[int]] = None,
                 animation: Optional[Animation] = None,
                 upload: Optional[UploadSettings] = None,
                 name: str = ""):
        self.num_frames = num_frames
        self.dtype = dtype
        self.shape = shape
        self.name = name
        self.animation = animation if animation is not None else FrameAnimation(boundary=AnimationBoundary.HOLD)
        self.upload = upload if upload is not None else UploadSettings(preupload=True)

        self.current_frame_index = 0

    def update(self, time: float, frame_index: int) -> T:
        self.current_frame_index = self.get_frame_index(time, frame_index)

    def get_current(self) -> T:
        return self.get_frame_by_index(self.current_frame_index)

    def max_animation_time(self, fps: float) -> float:
        a = self.animation.max_animation_time(self.num_frames, fps)
        return a

    def max_size() -> int:
        return 0

    def get_frame_index(self, time: float, playback_frame: int) -> int:
        return self.animation.get_frame_index(self.num_frames, time, playback_frame)

    def get_frame(self, time: float, frame_index: int) -> T:
        return self.get_frame_by_index(self.get_frame_index(time, frame_index))

    def get_frame_by_index_into(self, frame_index: int, out: memoryview, thread_index: int = -1) -> int:
        frame = self.get_frame_by_index(frame_index, thread_index)
        if frame is None:
            return 0
        data = view_bytes(frame)
        size = len(data)
        out[:size] = data
        return size

    def get_frame_by_index(self, frame_index: int, thread_index: int = -1) -> T:
        raise NotImplemented()


def view_bytes(a: np.ndarray) -> memoryview:
    return a.reshape((-1,), copy=False).view(np.uint8).data


def shape_match(expected_shape: Tuple[int], got_shape: Tuple[int]) -> bool:
    if expected_shape is None and got_shape is None:
        return True
    if len(expected_shape) != len(got_shape):
        return False
    for a, b in zip(expected_shape, got_shape):
        if a != -1 and a != b:
            return False
    return True


class DataProperty(Property):
    def __init__(self,
                 in_data: PropertyData,
                 dtype: Optional[np.dtype] = None,
                 shape: Optional[Tuple[int]] = None,
                 animation: Optional[Animation] = None,
                 upload: Optional[UploadSettings] = None,
                 name: str = ""):
        if shape is None:
            data = np.atleast_1d(np.asarray(in_data, dtype, order="C"))
            if len(data.shape) > 1:
                raise ShapeException((1,), data.shape[1:])
        else:
            if isinstance(in_data, List):
                data = in_data
                for i in range(len(in_data)):
                    a = np.asarray(in_data[i], dtype)
                    if not shape_match(shape, a.shape):
                        raise ShapeException(shape, a.shape)
                    data[i] = a
            else:
                data = np.asarray(in_data, dtype, order="C")
                if shape_match(shape, data.shape):
                    # Implicitly add animation dimension
                    data = data[np.newaxis]
                elif shape_match(shape, data.shape[1:]):
                    # Shape already matches
                    pass
                else:
                    raise ShapeException(shape, data.shape)

        super().__init__(len(data), dtype, shape, animation, upload, name)
        self.data: PropertyData = data

    def max_size(self) -> int:
        if isinstance(self.data, List):
            return max([d.itemsize * np.prod(d.shape) for d in self.data])
        else:
            return self.data.itemsize * np.prod(self.data.shape[1:])

    def get_frame_by_index(self, frame_index: int, thread_index: int = -1) -> T:
        return self.data[frame_index]


class StreamingProperty(Property):
    def __init__(self,
                 num_frames: int,
                 dtype: Optional[np.dtype] = None,
                 shape: Optional[Tuple[int]] = None,
                 animation: Optional[Animation] = None,
                 upload: Optional[UploadSettings] = None,
                 name: str = ""):
        super().__init__(num_frames, dtype, shape, animation, upload, name)

    def max_size(self) -> int:
        return NotImplemented()

    # def get_frame_by_index_into(self, frame_index: int, out: memoryview) -> int:
    def get_frame_by_index(self, frame: int, thread_index: int = -1) -> T:
        return NotImplemented()


class ShapeException(Exception):
    def __init__(self, expected: Tuple[int], got: Tuple[int]):
        self.expected = expected
        self.got = got

    def __str__(self):
        return f"Shape mismatch. Expected: {self.expected}. Got: {self.got}"

def as_property(value: Union[Property[T], PropertyData],
                dtype: Optional[np.dtype] = None,
                shape: Optional[Tuple[int]] = None,
                animation: Optional[Animation] = None,
                upload: Optional[UploadSettings] = None,
                name: str = ""):
    if isinstance(value, Property):
        if not shape_match(shape, value.shape):
            raise ShapeException(shape, value.shape)
        if not np.dtype(dtype) == value.dtype:
            raise TypeError(f"dtype mismatch. Expected: {dtype}. Got: {value.dtype}")
        if animation is not None:
            value.animation = animation
        if upload is not None:
            value.upload = upload
        if not value.name:
            value.name = name
        return value
    else:
        return DataProperty(value, dtype, shape, animation, upload, name)


_counter = 0

class Object:
    def __init__(self, name: Optional[str] = None):
        self.name = name or f"{type(self).__name__} - {Object.next_id()}"
        self.children: List["Object"] = []
        self.properties: List[Property] = []

        self.created = False

    @staticmethod
    def next_id():
        global _counter
        _counter += 1
        return _counter

    def add_property(self, prop: Property, dtype: Optional[np.dtype] = None, shape: Optional[Tuple[int]] = None, animation: Optional[Animation] = None, upload: Optional[UploadSettings] = None, name: str = "") -> Property:
        property = as_property(prop, dtype, shape, animation, upload, name)
        self.properties.append(property)
        return property

    def create_if_needed(self, renderer):
        if not self.created:
            self.create(renderer)
            self.created = True

    def create(self, renderer):
        pass

    def update(self, time: float, frame: int):
        # TODO: potentially the same property is updated more than once if added to multiple nodes
        # decide if this is ok because update will be idempotent or if we should dedup this.
        #
        # Currently update is idempotent because it's only doing frame lookup and caching of frame index / time
        for p in self.properties:
            p.update(time, frame)

    def update_transform(self, parent: "Object"):
        # TODO: can merge with udpate? Where to find parent? With link?
        pass

    def render(self, renderer, frame):
        pass

    def destroy(self):
        pass


class Object2D(Object):
    def __init__(self,
                 name: str,
                 translation: Optional[Property[vec2]] = None,
                 rotation: Optional[Property[float]] = None,
                 scale: Optional[Property[vec2]] = None):
        super().__init__(name)
        self.translation = self.add_property(translation if translation is not None else np.array([0, 0]), np.float32, (2,), name=f"translation")
        self.rotation = self.add_property(rotation if rotation is not None else np.array([0]), np.float32, name=f"rotation")
        self.scale = self.add_property(scale if scale is not None else np.array([1, 1]), np.float32, (2,), name=f"scale")
        self.update_transform(None)

    def update_transform(self, parent: "Object2D"):
        self.current_relative_transform = Transform2D (
            vec2(self.translation.get_current()),
            self.rotation.get_current(),
            vec2(self.scale.get_current())
        )
        self.current_transform_matrix = (parent.current_transform_matrix if parent is not None else np.eye(3)) @ self.current_relative_transform.as_mat3()

class Object3D(Object):
    def __init__(self,
                 name: str,
                 translation: Optional[Property[np.ndarray]] = None,
                 rotation: Optional[Property[np.ndarray]] = None,
                 scale: Optional[Property[np.ndarray]] = None):
        super().__init__(name)
        self.translation = self.add_property(translation if translation is not None else np.array([0, 0, 0]), np.float32, (3,), name=f"translation")
        self.rotation = self.add_property(rotation if rotation is not None else np.array([1, 0, 0, 0]), np.float32, (4,), name=f"rotation")
        self.scale = self.add_property(scale if scale is not None else np.array([1, 1, 1]), np.float32, (3,), name=f"scale")

        self.update_transform(None)

    def update_transform(self, parent: "Object3D"):
        self.current_relative_transform = Transform3D (
            vec3(self.translation.get_current()),
            quat(self.rotation.get_current()),
            vec3(self.scale.get_current())
        )
        self.current_transform_matrix = (parent.current_transform_matrix if parent is not None else np.eye(4)) @ self.current_relative_transform.as_mat4()

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

    def create_if_needed(self, renderer):
        def visit(p: Object, o: Object):
            o.create_if_needed(renderer)
        self.visit_objects(visit)

    def render(self, renderer, frame):
        def visit(p: Object, o: Object):
            o.render(renderer, frame)
        self.visit_objects(visit)