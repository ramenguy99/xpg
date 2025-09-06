from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import DTypeLike

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
