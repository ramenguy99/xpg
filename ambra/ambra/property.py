# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from pyxpg import Format

from .utils.gpu import format_from_channels_dtype, view_bytes
from . import gpu_property

PropertyItem = NDArray[Any]

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

    def map_frame_index(self, frame: int, n: int) -> int:
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
            raise ValueError(f"Unhandled enum variant: {self}")

    def map_frame_index_range(self, frame: int, begin: int, end: int) -> int:
        if self == AnimationBoundary.HOLD:
            return np.clip(frame, begin, end - 1)  # type: ignore

        n = end - begin
        norm_frame = frame - begin
        if self == AnimationBoundary.MIRROR:
            dr = norm_frame % (2 * n)
            if dr >= n:
                return 2 * n - dr - 1 + begin
            else:
                return dr + begin
        if self == AnimationBoundary.REPEAT:
            return (norm_frame % n) + begin
        else:
            raise ValueError(f"Unhandled enum variant: {self}")

    def map_time(self, t: float, t_min: float, t_max: float) -> float:
        if self == AnimationBoundary.HOLD:
            return np.clip(t, t_min, t_max)  # type: ignore
        elif self == AnimationBoundary.MIRROR:
            double_dt = (t_max - t_min) * 2.0
            dr: float = np.fmod(t - t_min, double_dt)
            if dr >= t_max - t_min:
                return double_dt - dr + t_min
            else:
                return dr + t_min
        elif self == AnimationBoundary.REPEAT:
            return np.fmod(t - t_min, t_max - t_min) + t_min  # type: ignore
        else:
            raise ValueError(f"Unhandled enum variant: {self}")


class Animation:
    # interpolation = AnimationInterpolation.NEAREST
    def __init__(self, boundary: AnimationBoundary):
        self.boundary = boundary

    def get_frame_index(self, n: int, time: float, frame_index: int) -> int:
        return 0

    def max_animation_time(self, n: int, fps: float) -> float:
        return 0


class TimeSampledAnimation(Animation):
    def __init__(self, boundary: AnimationBoundary, timestamps: NDArray[np.float64]):
        super().__init__(boundary)
        self.timestamps = np.asarray(timestamps)

    def get_frame_index(self, n: int, time: float, frame_index: int) -> int:
        count = self.timestamps.size
        if count == 0:
            return -1
        elif count == 1:
            return 0
        else:
            t = self.boundary.map_time(time, self.timestamps[0], self.timestamps[-1])
            return max(np.searchsorted(self.timestamps, t, side="right") - 1, 0)  # type: ignore

    def max_animation_time(self, n: int, fps: float) -> float:
        return self.timestamps[-1] if self.timestamps.size > 0 else 0.0


@dataclass
class FrameSampledAnimation(Animation):
    def __init__(self, boundary: AnimationBoundary, indices: NDArray[np.uint32]):
        super().__init__(boundary)
        self.indices = np.asarray(indices, np.uint32)

    def get_frame_index(self, n: int, time: float, frame_index: int) -> int:
        count = self.indices.size
        if count == 0:
            return -1
        elif count == 1:
            return 0
        else:
            idx = self.boundary.map_frame_index_range(frame_index, self.indices[0], self.indices[-1] + 1)
            return max(np.searchsorted(self.indices, idx, side="right") - 1, 0)  # type: ignore

    def max_animation_time(self, n: int, fps: float) -> float:
        return (self.indices[-1] + 1) / fps if self.indices.size > 0 else 0.0


@dataclass
class ConstantSpeedAnimation(Animation):
    frames_per_second: float  # frame time in seconds

    def get_frame_index(self, n: int, time: float, frame_index: int) -> int:
        if n == 0:
            return -1
        return self.boundary.map_frame_index(int(time * self.frames_per_second), n)

    def max_animation_time(self, n: int, fps: float) -> float:
        return n / self.frames_per_second


class FrameAnimation(Animation):
    def get_frame_index(self, n: int, time: float, frame_index: int) -> int:
        if n == 0:
            return -1
        return self.boundary.map_frame_index(frame_index, n)

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
        animation: Optional[Animation] = None,
        upload: Optional[UploadSettings] = None,
        name: str = "",
    ):
        self.num_frames = num_frames
        self.name = name
        self.animation = animation if animation is not None else FrameAnimation(boundary=AnimationBoundary.REPEAT)
        self.upload = upload if upload is not None else UploadSettings(preupload=True)

        self.current_frame_index = 0

        self.update_callbacks: List[Callable[[Property], None]] = []

    def update(self, time: float, playback_frame: int) -> None:
        old_frame = self.current_frame_index
        self.current_frame_index = self.get_frame_index(time, playback_frame)

        if old_frame != self.current_frame_index:
            for c in self.update_callbacks:
                c(self)

    def get_current(self) -> PropertyItem:
        return self.get_frame_by_index(self.current_frame_index)

    def max_animation_time(self, fps: float) -> float:
        return self.animation.max_animation_time(self.num_frames, fps)

    def get_frame_index(self, time: float, playback_frame: int) -> int:
        return self.animation.get_frame_index(self.num_frames, time, playback_frame)

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


class BufferProperty(Property):
    def __init__(
        self,
        max_size: int,
        num_frames: int,
        dtype: DTypeLike,
        shape: Tuple[int, ...],
        animation: Optional[Animation] = None,
        upload: Optional[UploadSettings] = None,
        name: str = "",
    ):
        super().__init__(num_frames, animation, upload, name)
        self.max_size = max_size
        self.dtype = dtype
        self.shape = shape
        self.gpu_property: Optional[gpu_property.GpuBufferProperty] = None


class ImageProperty(Property):
    def __init__(
        self,
        width: int,
        height: int,
        format: Format,
        num_frames: int,
        animation: Optional[Animation] = None,
        upload: Optional[UploadSettings] = None,
        name: str = "",
    ):
        super().__init__(num_frames, animation, upload, name)
        self.width = width
        self.height = height
        self.format = format
        self.gpu_property: Optional[gpu_property.GpuImageProperty] = None


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


class ArrayBufferProperty(BufferProperty):
    def __init__(
        self,
        data: ArrayLike,
        dtype: Optional[DTypeLike] = None,
        animation: Optional[Animation] = None,
        upload: Optional[UploadSettings] = None,
        name: str = "",
    ):
        data = np.atleast_1d(np.asarray(data, dtype, order="C"))
        max_size = data.itemsize * np.prod(data.shape[1:], dtype=np.uint64)  # type: ignore
        super().__init__(max_size, len(data), data.dtype, data.shape[1:], animation, upload, name)
        self.data = data

    def get_frame_by_index(self, frame_index: int, thread_index: int = -1) -> PropertyItem:
        return self.data[frame_index]


class ListBufferProperty(BufferProperty):
    def __init__(
        self,
        data: List[ArrayLike],
        dtype: Optional[DTypeLike] = None,
        shape: Optional[Tuple[int, ...]] = None,
        max_size: int = 0,
        animation: Optional[Animation] = None,
        upload: Optional[UploadSettings] = None,
        name: str = "",
    ):
        property_data = []
        for i in range(len(data)):
            a = np.asarray(data[i], dtype, order="C")

            # Shape checking
            if not shape_match(shape, a.shape):
                raise ShapeError(shape, a.shape)

            # If no dtype was given, use dtype inferred for first element
            if dtype is None:
                dtype = a.dtype

            property_data.append(a)
            max_size = max(max_size, a.nbytes)

        super().__init__(max_size, len(property_data), dtype, shape, animation, upload, name)
        self.data = property_data

    def get_frame_by_index(self, frame_index: int, thread_index: int = -1) -> PropertyItem:
        return self.data[frame_index]


class ArrayImageProperty(ImageProperty):
    def __init__(
        self,
        in_data: ArrayLike,
        format: Optional[Format] = None,
        animation: Optional[Animation] = None,
        upload: Optional[UploadSettings] = None,
        name: str = "",
    ):
        data = np.asarray(in_data, order="C")
        if len(data.shape) != 4:
            raise ShapeError((-1, -1, -1), data.shape)

        height, width, channels = data.shape[1:4]
        if format is None:
            format = format_from_channels_dtype(channels, data.dtype)
        else:
            # TODO: check that format is compatible with data (maybe allow some conversions?)
            pass

        super().__init__(width, height, format, len(data), animation, upload, name)
        self.data = data

    def get_frame_by_index(self, frame_index: int, thread_index: int = -1) -> PropertyItem:
        return self.data[frame_index]

class ListImageProperty(ImageProperty):
    def __init__(
        self,
        data: List[ArrayLike],
        format: Optional[Format] = None,
        animation: Optional[Animation] = None,
        upload: Optional[UploadSettings] = None,
        name: str = "",
    ):
        property_data: List[NDArray[Any]] = []
        property_shape: Optional[Tuple[int, int, int]] = None
        property_dtype: DTypeLike = None
        for i in range(len(data)):
            a = np.asarray(data[i], property_dtype, order="C")
            if len(a.shape) != 3:
                raise ShapeError((-1, -1, -1), a.shape)
            if property_shape is None:
                property_shape = a.shape
            else:
                if property_shape != a.shape:
                    raise RuntimeError(f"ListImageProperty data elements must all have the same shape. First has {property_shape}. Element {i} has {a.shape}")
            if property_dtype is None:
                property_dtype = a.dtype
            property_data.append(a)

        height, width, channels = property_data[0].shape[:3]
        if format is None:
            format = format_from_channels_dtype(channels, property_data[0].dtype)
        else:
            # TODO: check that format is compatible with data (maybe allow some conversions?)
            pass

        super().__init__(width, height, format, len(property_data), animation, upload, name)
        self.data = property_data

    def get_frame_by_index(self, frame_index: int, thread_index: int = -1) -> PropertyItem:
        return self.data[frame_index]


def as_buffer_property(
    value: Union[BufferProperty, List, ArrayLike],
    dtype: Optional[DTypeLike] = None,
    shape: Optional[Tuple[int, ...]] = None,
    name: str = "",
) -> BufferProperty:
    if isinstance(value, BufferProperty):
        if not shape_match(shape, value.shape):
            raise ShapeError(shape, value.shape)
        if dtype is not None and not np.dtype(dtype) == value.dtype:
            raise TypeError(f"dtype mismatch. Expected: {dtype}. Got: {value.dtype}")
        if not value.name:
            value.name = name
        return value
    elif isinstance(value, List):
        return ListBufferProperty(value, dtype, shape, name)
    else:
        if shape is None:
            value = np.atleast_1d(np.asarray(value, dtype, order="C"))
            if len(value.shape) > 1:
                raise ShapeError((1,), value.shape[1:])

        if shape_match(shape, value.shape):
            # Implicitly add animation dimension
            value = value[np.newaxis]
        elif shape_match(shape, value.shape[1:]):
            # Shape already matches
            pass
        else:
            raise ShapeError(shape, value.shape)

        return ArrayBufferProperty(value, dtype, name)


def as_image_property(
    value: Union[ImageProperty, List, ArrayLike],
    name: str = "",
) -> ImageProperty:
    if isinstance(value, ImageProperty):
        if not value.name:
            value.name = name
        return value
    elif isinstance(value, List):
        return ListImageProperty(value, None, name=name)
    else:
        if value.ndim == 3:
            # Implicitly add animation dimension
            value = value[np.newaxis]
        elif value.ndim == 4:
            # Shape already matches
            pass
        else:
            raise ShapeError((-1, -1, -1), value.shape)

        return ArrayImageProperty(value, None, name=name)
