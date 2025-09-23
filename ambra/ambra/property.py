# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import DTypeLike
from pyxpg import Format

from .utils.gpu import format_from_channels_dtype

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


def view_bytes(a: np.ndarray) -> memoryview:
    return a.reshape((-1,), copy=False).view(np.uint8).data


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
        animation: Optional[Animation] = None,
        upload: Optional[UploadSettings] = None,
        name: str = "",
    ):
        self.num_frames = num_frames
        self.name = name
        self.animation = animation if animation is not None else FrameAnimation(boundary=AnimationBoundary.REPEAT)
        self.upload = upload if upload is not None else UploadSettings(preupload=True)

        self.current_frame_index = 0

    def update(self, time: float, playback_frame: int) -> None:
        self.current_frame_index = self.get_frame_index(time, playback_frame)

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
        dtype: Optional[DTypeLike] = None,
        shape: Optional[Tuple[int, ...]] = None,
        animation: Optional[Animation] = None,
        upload: Optional[UploadSettings] = None,
        name: str = "",
    ):
        super().__init__(num_frames, animation, upload, name)
        self.max_size = max_size
        self.dtype = dtype
        self.shape = shape


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


class DataBufferProperty(BufferProperty):
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
                    a = np.asarray(in_data[i], dtype, order="C")
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

        max_size: int
        if isinstance(data, List):
            max_size = max([d.itemsize * np.prod(d.shape, dtype=np.int64) for d in data])  # type: ignore
        else:
            max_size = data.itemsize * np.prod(data.shape[1:], dtype=np.int64)  # type: ignore

        super().__init__(max_size, len(data), dtype, shape, animation, upload, name)
        self.data = data

    def get_frame_by_index(self, frame_index: int, thread_index: int = -1) -> PropertyItem:
        return self.data[frame_index]


class DataImageProperty(ImageProperty):
    def __init__(
        self,
        in_data: PropertyData,
        format: Optional[Format] = None,
        animation: Optional[Animation] = None,
        upload: Optional[UploadSettings] = None,
        name: str = "",
    ):
        data: Union[List[np.ndarray], np.ndarray]
        if isinstance(in_data, List):
            data = in_data
            for i in range(len(in_data)):
                a = np.asarray(in_data[i])
                if len(a.shape) != 3:
                    raise ShapeError((-1, -1, -1), a.shape)
                data[i] = a
        else:
            data = np.asarray(in_data, order="C")
            if len(data.shape) == 3:
                # Implicitly add animation dimension
                data = data[np.newaxis]
            elif len(data.shape) == 4:
                # Shape already matches
                pass
            else:
                raise ShapeError((-1, -1, -1), data.shape)

        height, width, channels = data[0].shape[:3]
        if format is None:
            format = format_from_channels_dtype(channels, data[0].dtype)
        else:
            # TODO: check that format is compatible with data (maybe allow some conversions?)
            pass

        super().__init__(width, height, format, len(data), animation, upload, name)
        self.data = data

    def get_frame_by_index(self, frame_index: int, thread_index: int = -1) -> PropertyItem:
        return self.data[frame_index]


def as_buffer_property(
    value: Union[BufferProperty, PropertyData, int, float],
    dtype: Optional[DTypeLike] = None,
    shape: Optional[Tuple[int, ...]] = None,
    animation: Optional[Animation] = None,
    upload: Optional[UploadSettings] = None,
    name: str = "",
) -> BufferProperty:
    if isinstance(value, BufferProperty):
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
        return DataBufferProperty(value, dtype, shape, animation, upload, name)


def as_image_property(
    value: Union[ImageProperty, PropertyData],
    animation: Optional[Animation] = None,
    upload: Optional[UploadSettings] = None,
    name: str = "",
) -> ImageProperty:
    if isinstance(value, ImageProperty):
        if animation is not None:
            value.animation = animation
        if upload is not None:
            value.upload = upload
        if not value.name:
            value.name = name
        return value
    else:
        return DataImageProperty(value, None, animation, upload, name)
