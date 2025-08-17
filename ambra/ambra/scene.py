from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import DTypeLike
from pyglm.glm import mat3, mat4, quat, vec2, vec3
from pyxpg import imgui

from .transform2d import Transform2D
from .transform3d import Transform3D

PropertyItem = np.ndarray
PropertyData = Union[List[np.ndarray], np.ndarray]

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
    frames_per_second: float  # frame time in seconds

    def get_frame_index(self, n: int, time: float, frame_index: int) -> int:
        if n == 0:
            return -1
        return self.boundary.map(int(time * self.frames_per_second), n)

    def max_animation_time(self, n: int, fps: float) -> float:
        return n / self.frames_per_second


@dataclass
class FrameAnimation(Animation):
    def get_frame_index(self, n: int, time: float, frame_index: int) -> int:
        if n == 0:
            return -1
        return self.boundary.map(frame_index, n)

    def max_animation_time(self, n: int, frames_per_second: float) -> float:
        return n / frames_per_second


@dataclass
class UploadSettings:
    preupload: bool
    async_load: bool = False
    cpu_prefetch_count: int = 0
    gpu_prefetch_count: int = 0


class Property:
    def __init__(
        self,
        num_frames: int,
        dtype: Optional[DTypeLike] = None,
        shape: Optional[Tuple[int, ...]] = None,
        animation: Optional[Animation] = None,
        upload: Optional[UploadSettings] = None,
        name: str = "",
    ):
        self.num_frames = num_frames
        self.dtype = dtype
        self.shape = shape
        self.name = name
        self.animation = animation if animation is not None else FrameAnimation(boundary=AnimationBoundary.REPEAT)
        self.upload = upload if upload is not None else UploadSettings(preupload=True)

        self.current_frame_index = 0

    def update(self, time: float, frame_index: int) -> None:
        self.current_frame_index = self.get_frame_index(time, frame_index)

    def get_current(self) -> PropertyItem:
        return self.get_frame_by_index(self.current_frame_index)

    def max_animation_time(self, fps: float) -> float:
        return self.animation.max_animation_time(self.num_frames, fps)

    def max_size(self) -> int:
        raise NotImplementedError

    def width(self) -> int:
        raise NotImplementedError

    def height(self) -> int:
        raise NotImplementedError

    def channels(self) -> int:
        raise NotImplementedError

    def get_frame_index(self, time: float, playback_frame: int) -> int:
        return self.animation.get_frame_index(self.num_frames, time, playback_frame)

    def get_frame(self, time: float, frame_index: int) -> PropertyItem:
        return self.get_frame_by_index(self.get_frame_index(time, frame_index))

    def get_frame_by_index_into(self, frame_index: int, out: memoryview, thread_index: int = -1) -> int:
        frame = self.get_frame_by_index(frame_index, thread_index)
        if frame is None:
            return 0
        data = view_bytes(frame)
        size = len(data)
        out[:size] = data
        return size

    def get_frame_by_index(self, frame_index: int, thread_index: int = -1) -> PropertyItem:
        raise NotImplementedError


def view_bytes(a: np.ndarray) -> memoryview:
    return a.reshape((-1,), copy=False).view(np.uint8).data


def shape_match(
    expected_shape: Optional[Tuple[int, ...]],
    got_shape: Optional[Tuple[int, ...]],
) -> bool:
    if expected_shape is None:
        return got_shape is None
    else:
        if got_shape is None:
            return False

    if len(expected_shape) != len(got_shape):
        return False
    for a, b in zip(expected_shape, got_shape):
        if a != -1 and a != b:
            return False
    return True


class DataProperty(Property):
    def __init__(
        self,
        in_data: Union[PropertyData, int, float],
        dtype: Optional[DTypeLike] = None,
        shape: Optional[Tuple[int, ...]] = None,
        animation: Optional[Animation] = None,
        upload: Optional[UploadSettings] = None,
        name: str = "",
    ):
        data: Union[List[np.ndarray], np.ndarray]
        if shape is None:
            data = np.atleast_1d(np.asarray(in_data, dtype, order="C"))
            if len(data.shape) > 1:
                raise ShapeError((1,), data.shape[1:])
        else:
            if isinstance(in_data, List):
                data = in_data
                for i in range(len(in_data)):
                    a = np.asarray(in_data[i], dtype)
                    if not shape_match(shape, a.shape):
                        raise ShapeError(shape, a.shape)
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
                    raise ShapeError(shape, data.shape)

        super().__init__(len(data), dtype, shape, animation, upload, name)
        self.data: PropertyData = data

    # Buffers
    def max_size(self) -> int:
        if isinstance(self.data, List):
            return max([d.itemsize * np.prod(d.shape, dtype=np.int64) for d in self.data])  # type: ignore
        else:
            return self.data.itemsize * np.prod(self.data.shape[1:], dtype=np.int64)  # type: ignore

    # Images
    def width(self) -> int:
        return self.data[0].shape[1]  # type: ignore

    def height(self) -> int:
        return self.data[0].shape[0]  # type: ignore

    def channels(self) -> int:
        return self.data[0].shape[2]  # type: ignore

    def get_frame_by_index(self, frame_index: int, thread_index: int = -1) -> PropertyItem:
        return self.data[frame_index]


class StreamingProperty(Property):
    def __init__(
        self,
        num_frames: int,
        dtype: Optional[DTypeLike] = None,
        shape: Optional[Tuple[int, ...]] = None,
        animation: Optional[Animation] = None,
        upload: Optional[UploadSettings] = None,
        name: str = "",
    ):
        super().__init__(num_frames, dtype, shape, animation, upload, name)

    # For buffers
    def max_size(self) -> int:
        raise NotImplementedError

    # For images
    def width(self) -> int:
        raise NotImplementedError

    def height(self) -> int:
        raise NotImplementedError

    def channels(self) -> int:
        raise NotImplementedError

    # def get_frame_by_index_into(self, frame_index: int, out: memoryview) -> int:
    def get_frame_by_index(self, frame: int, thread_index: int = -1) -> PropertyItem:
        raise NotImplementedError


class ShapeError(Exception):
    def __init__(
        self,
        expected: Optional[Tuple[int, ...]],
        got: Optional[Tuple[int, ...]],
    ):
        self.expected = expected
        self.got = got

    def __str__(self) -> str:
        return f"Shape mismatch. Expected: {self.expected}. Got: {self.got}"


def as_property(
    value: Union[Property, PropertyData, int, float],
    dtype: Optional[DTypeLike] = None,
    shape: Optional[Tuple[int, ...]] = None,
    animation: Optional[Animation] = None,
    upload: Optional[UploadSettings] = None,
    name: str = "",
) -> Property:
    if isinstance(value, Property):
        if not shape_match(shape, value.shape):
            raise ShapeError(shape, value.shape)
        if dtype is not None and not np.dtype(dtype) == value.dtype:
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
        self.uid = Object.next_id()
        self.name = name or f"{type(self).__name__} - {self.uid}"
        self.children: List[Object] = []
        self.properties: List[Property] = []

        self.created = False
        self.gui_expanded = False
        self.gui_selected = False
        self.gui_selected_property: Optional[Property] = None

    @staticmethod
    def next_id() -> int:
        global _counter
        _counter += 1
        return _counter

    def add_property(
        self,
        prop: Union[Property, PropertyData, int, float],
        dtype: Optional[DTypeLike] = None,
        shape: Optional[Tuple[int, ...]] = None,
        animation: Optional[Animation] = None,
        upload: Optional[UploadSettings] = None,
        name: str = "",
    ) -> Property:
        property = as_property(prop, dtype, shape, animation, upload, name)
        self.properties.append(property)
        return property

    def create_if_needed(self, renderer):  # type: ignore
        if not self.created:
            self.create(renderer)  # type: ignore
            self.created = True

    def create(self, renderer):  # type: ignore
        pass

    def update(self, time: float, frame: int) -> None:
        # TODO: potentially the same property is updated more than once if added to multiple nodes
        # decide if this is ok because update will be idempotent or if we should dedup this.
        #
        # Currently update is idempotent because it's only doing frame lookup and caching of frame index / time
        for p in self.properties:
            p.update(time, frame)

    def update_transform(self, parent: Optional["Object"]) -> None:
        # TODO: can merge with udpate? Where to find parent? With link?
        pass

    def render(self, renderer, frame):  # type: ignore
        pass

    def destroy(self) -> None:
        pass

    def gui(self) -> None:
        imgui.text("Properties:")
        imgui.indent(5)
        for p in self.properties:
            s = imgui.selectable(f"{p.name}", self.gui_selected_property == p)
            if s:
                self.gui_selected_property = p
            if self.gui_selected_property == p:
                imgui.indent(10)
                imgui.text(f"{p.shape} {np.dtype(p.dtype).name} {p.max_size()}")
                if p.num_frames > 1:
                    imgui.text(f"{p.current_frame_index} / {p.num_frames}")
                imgui.indent(-10)
        imgui.indent(-5)


class Object2D(Object):
    def __init__(
        self,
        name: Optional[str],
        translation: Optional[Property] = None,
        rotation: Optional[Property] = None,
        scale: Optional[Property] = None,
    ):
        super().__init__(name)
        self.translation = self.add_property(
            translation if translation is not None else np.array([0, 0]),
            np.float32,
            (2,),
            name="translation",
        )
        self.rotation = self.add_property(
            rotation if rotation is not None else np.array([0]),
            np.float32,
            name="rotation",
        )
        self.scale = self.add_property(
            scale if scale is not None else np.array([1, 1]),
            np.float32,
            (2,),
            name="scale",
        )
        self.update_transform(None)

    def update_transform(self, parent: Optional[Object]) -> None:
        assert parent is None or isinstance(parent, Object2D)

        self.current_relative_transform = Transform2D(
            vec2(self.translation.get_current()),
            self.rotation.get_current().item(),
            vec2(self.scale.get_current()),
        )
        self.current_transform_matrix: mat3 = (
            parent.current_transform_matrix if parent is not None else np.eye(3)
        ) @ self.current_relative_transform.as_mat3()  # type: ignore


class Object3D(Object):
    def __init__(
        self,
        name: Optional[str] = None,
        translation: Optional[Property] = None,
        rotation: Optional[Property] = None,
        scale: Optional[Property] = None,
    ):
        super().__init__(name)
        self.translation = self.add_property(
            translation if translation is not None else np.array([0, 0, 0]),
            np.float32,
            (3,),
            name="translation",
        )
        self.rotation = self.add_property(
            rotation if rotation is not None else np.array([1, 0, 0, 0]),
            np.float32,
            (4,),
            name="rotation",
        )
        self.scale = self.add_property(
            scale if scale is not None else np.array([1, 1, 1]),
            np.float32,
            (3,),
            name="scale",
        )

        self.update_transform(None)

    def update_transform(self, parent: Optional[Object]) -> None:
        assert parent is None or isinstance(parent, Object3D)

        self.current_relative_transform = Transform3D(
            vec3(self.translation.get_current()),
            quat(self.rotation.get_current()),
            vec3(self.scale.get_current()),
        )
        self.current_transform_matrix: mat4 = (
            parent.current_transform_matrix if parent is not None else np.eye(4)
        ) @ self.current_relative_transform.as_mat4()  # type: ignore


class Scene:
    def __init__(self, name: str):
        self.name = name
        self.objects: List[Object] = []

    def visit_objects_with_parent(self, function: Callable[[Optional[Object], Object], None]) -> None:
        def visit_recursive(p: Optional[Object], o: Object) -> None:
            function(p, o)
            for c in o.children:
                visit_recursive(o, c)

        for o in self.objects:
            visit_recursive(None, o)

    def visit_objects(self, function: Callable[[Object], None]) -> None:
        def visit_recursive(o: Object) -> None:
            function(o)
            for c in o.children:
                visit_recursive(c)

        for o in self.objects:
            visit_recursive(o)

    def visit_objects_pre_post(self, pre: Callable[[Object], bool], post: Callable[[Object], None]) -> None:
        def visit_recursive(o: Object) -> None:
            rec = pre(o)
            if rec:
                for c in o.children:
                    visit_recursive(c)
                post(o)

        for o in self.objects:
            visit_recursive(o)

    def max_animation_time(self, frames_per_second: float) -> float:
        time = 0.0

        def visit(o: Object) -> None:
            nonlocal time
            time = max(
                time,
                max(p.max_animation_time(frames_per_second) for p in o.properties),
            )

        self.visit_objects(visit)
        return time

    def update(self, time: float, frame: int) -> None:
        def visit(p: Optional[Object], o: Object) -> None:
            o.update(time, frame)
            o.update_transform(p)

        self.visit_objects_with_parent(visit)

    def create_if_needed(self, renderer) -> None:  # type: ignore
        def visit(o: Object) -> None:
            o.create_if_needed(renderer)  # type: ignore

        self.visit_objects(visit)

    def render(self, renderer, frame) -> None:  # type: ignore
        def visit(o: Object) -> None:
            o.render(renderer, frame)  # type: ignore

        self.visit_objects(visit)
