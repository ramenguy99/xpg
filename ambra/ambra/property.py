# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

import bisect
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from pyxpg import (
    BufferUsageFlags,
    Format,
    ImageLayout,
    ImageUsageFlags,
    PipelineStageFlags,
)

from . import gpu_property as gpu_property_mod
from .utils.gpu import format_from_channels_dtype, view_bytes

if TYPE_CHECKING:
    from .renderer import Renderer


PropertyItem = NDArray[Any]


class AnimationBoundary(Enum):
    HOLD = auto()
    REPEAT = auto()
    MIRROR = auto()
    DISABLE = auto()


class Animation:
    def __init__(self, boundary: AnimationBoundary = AnimationBoundary.HOLD):
        self.boundary = boundary

    def get_frame_index(self, n: int, time: float, frame_index: int) -> int:
        return -1

    def is_dynamic(self, n: int) -> bool:
        return False

    def is_disabled(self, n: int, time: float, frame_index: int) -> bool:
        return True

    def start_animation_time(self, n: int, fps: float) -> float:
        return 0

    def end_animation_time(self, n: int, fps: float) -> float:
        return 0

    def is_frame_index_disabled(self, frame: int, begin: int, end: int) -> bool:
        return (frame < begin) or (self.boundary == AnimationBoundary.DISABLE and frame >= end)

    def is_time_disabled(self, t: float, t_min: float, t_max: float) -> bool:
        return (t < t_min) or (self.boundary == AnimationBoundary.DISABLE and t >= t_max)

    def map_frame_index(self, frame: int, begin: int, end: int) -> int:
        if self.boundary == AnimationBoundary.HOLD or self.boundary == AnimationBoundary.DISABLE:
            return int(np.clip(frame, begin, end - 1))

        n = end - begin
        norm_frame = frame - begin
        if self.boundary == AnimationBoundary.REPEAT:
            return (norm_frame % n) + begin
        elif self.boundary == AnimationBoundary.MIRROR:
            dr = norm_frame % (2 * n)
            if dr >= n:
                return 2 * n - dr - 1 + begin
            else:
                return dr + begin
        else:
            raise ValueError(f"Unhandled enum variant: {self.boundary}")

    def map_time(self, t: float, t_min: float, t_max: float) -> float:
        if self.boundary == AnimationBoundary.HOLD or self.boundary == AnimationBoundary.DISABLE:
            return np.clip(t, t_min, t_max)  # type: ignore
        elif self.boundary == AnimationBoundary.REPEAT:
            return np.fmod(t - t_min, t_max - t_min) + t_min  # type: ignore
        elif self.boundary == AnimationBoundary.MIRROR:
            double_dt = (t_max - t_min) * 2.0
            dr: float = np.fmod(t - t_min, double_dt)
            if dr >= t_max - t_min:
                return double_dt - dr + t_min
            else:
                return dr + t_min
        else:
            raise ValueError(f"Unhandled enum variant: {self.boundary}")


class TimeSampledAnimation(Animation):
    def __init__(self, timestamps: NDArray[np.float64], boundary: AnimationBoundary = AnimationBoundary.HOLD):
        super().__init__(boundary)
        self.timestamps = np.asarray(timestamps)

    def get_frame_index(self, n: int, time: float, frame_index: int) -> int:
        count = self.timestamps.size
        if count == 0:
            return -1
        elif count == 1:
            return 0
        else:
            t = self.map_time(time, self.timestamps[0], self.timestamps[-1])
            return max(int(np.searchsorted(self.timestamps, t, side="right")) - 1, 0)

    def is_dynamic(self, n: int) -> bool:
        return True

    def is_disabled(self, n: int, time: float, frame_index: int) -> bool:
        count = self.timestamps.size
        if count == 0:
            return True
        else:
            return self.is_time_disabled(time, self.timestamps[0], self.timestamps[-1])

    def start_animation_time(self, n: int, fps: float) -> float:
        return self.timestamps[0] if self.timestamps else 0.0

    def end_animation_time(self, n: int, fps: float) -> float:
        return self.timestamps[-1] if self.timestamps.size > 0 else 0.0


class ListTimeSampledAnimation(Animation):
    def __init__(self, timestamps: List[float], boundary: AnimationBoundary = AnimationBoundary.HOLD):
        super().__init__(boundary)
        self.timestamps = timestamps

    def get_frame_index(self, n: int, time: float, frame_index: int) -> int:
        count = len(self.timestamps)
        if count == 0:
            return -1
        elif count == 1:
            return 0
        else:
            t = self.map_time(time, self.timestamps[0], self.timestamps[-1])
            return max(bisect.bisect_right(self.timestamps, t) - 1, 0)

    def is_dynamic(self, n: int) -> bool:
        return True

    def is_disabled(self, n: int, time: float, frame_index: int) -> bool:
        if not self.timestamps:
            return True
        else:
            return self.is_time_disabled(time, self.timestamps[0], self.timestamps[-1])

    def start_animation_time(self, n: int, fps: float) -> float:
        return self.timestamps[0] if self.timestamps else 0.0

    def end_animation_time(self, n: int, fps: float) -> float:
        return self.timestamps[-1] if self.timestamps else 0.0


class FrameSampledAnimation(Animation):
    def __init__(self, indices: NDArray[np.uint32], boundary: AnimationBoundary = AnimationBoundary.HOLD):
        super().__init__(boundary)
        self.indices = np.asarray(indices, np.uint32)

    def get_frame_index(self, n: int, time: float, frame_index: int) -> int:
        count = self.indices.size
        if count == 0:
            return -1
        elif count == 1:
            return 0
        else:
            idx = self.map_frame_index(frame_index, self.indices[0], self.indices[-1] + 1)
            return max(int(np.searchsorted(self.indices, idx, side="right")) - 1, 0)

    def is_dynamic(self, n: int) -> bool:
        return True

    def is_disabled(self, n: int, time: float, frame_index: int) -> bool:
        count = self.indices.size
        if count == 0:
            return True
        else:
            return self.is_frame_index_disabled(frame_index, self.indices[0], self.indices[-1])

    def start_animation_time(self, n: int, fps: float) -> float:
        return self.indices[0] / fps if self.indices else 0.0

    def end_animation_time(self, n: int, fps: float) -> float:
        return (self.indices[-1] + 1) / fps if self.indices.size > 0 else 0.0


class ListFrameSampledAnimation(Animation):
    def __init__(self, indices: List[int], boundary: AnimationBoundary = AnimationBoundary.HOLD):
        super().__init__(boundary)
        self.indices = indices

    def get_frame_index(self, n: int, time: float, frame_index: int) -> int:
        count = len(self.indices)
        if count == 0:
            return -1
        elif count == 1:
            return 0
        else:
            idx = self.map_frame_index(frame_index, self.indices[0], self.indices[-1] + 1)
            return max(bisect.bisect_right(self.indices, idx) - 1, 0)

    def is_dynamic(self, n: int) -> bool:
        return True

    def is_disabled(self, n: int, time: float, frame_index: int) -> bool:
        if not self.indices:
            return True
        else:
            return self.is_frame_index_disabled(frame_index, self.indices[0], self.indices[-1])

    def start_animation_time(self, n: int, fps: float) -> float:
        return self.indices[0] / fps if self.indices else 0.0

    def end_animation_time(self, n: int, fps: float) -> float:
        return (self.indices[-1] + 1) / fps if self.indices else 0.0


class ConstantSpeedAnimation(Animation):
    def __init__(
        self, frames_per_second: float, start_frame: int = 0, boundary: AnimationBoundary = AnimationBoundary.HOLD
    ):
        super().__init__(boundary)
        self.frames_per_second = frames_per_second
        self.start_frame = start_frame

    def get_frame_index(self, n: int, time: float, frame_index: int) -> int:
        if n == 0:
            return -1
        return self.map_frame_index(int(time * self.frames_per_second), 0, n)

    def is_dynamic(self, n: int) -> bool:
        return n > 1 or self.start_frame > 0

    def is_disabled(self, n: int, time: float, frame_index: int) -> bool:
        if n == 0:
            return True
        else:
            return self.is_frame_index_disabled(int(time * self.frames_per_second), 0, n)

    def start_animation_time(self, n: int, fps: float) -> float:
        return self.start_frame / self.frames_per_second

    def end_animation_time(self, n: int, fps: float) -> float:
        return (self.start_frame + n) / self.frames_per_second


class FrameAnimation(Animation):
    def __init__(self, start_frame: int = 0, boundary: AnimationBoundary = AnimationBoundary.HOLD):
        super().__init__(boundary)
        self.start_frame = start_frame

    def get_frame_index(self, n: int, time: float, frame_index: int) -> int:
        if n == 0 or frame_index < self.start_frame:
            return -1
        return self.map_frame_index(frame_index - self.start_frame, 0, n)

    def is_dynamic(self, n: int) -> bool:
        return n > 1 or self.start_frame > 0

    def is_disabled(self, n: int, time: float, frame_index: int) -> bool:
        if n == 0 or frame_index < self.start_frame:
            return True
        else:
            return self.is_frame_index_disabled(frame_index - self.start_frame, 0, n)

    def start_animation_time(self, n: int, frames_per_second: float) -> float:
        return self.start_frame / frames_per_second

    def end_animation_time(self, n: int, frames_per_second: float) -> float:
        return (self.start_frame + n) / frames_per_second


@dataclass
class UploadSettings:
    preupload: bool = True
    batched: bool = True
    async_load: bool = False
    cpu_prefetch_count: int = 0
    gpu_prefetch_count: int = 0


class Property:
    gpu_property: Optional[gpu_property_mod.GpuProperty[Any]]

    def __init__(
        self,
        num_frames: int,
        animation: Optional[Animation] = None,
        upload: Optional[UploadSettings] = None,
        name: str = "",
    ):
        self.num_frames = num_frames
        self.name = name
        self.animation = animation if animation is not None else FrameAnimation()
        self.upload = upload if upload is not None else UploadSettings()

        self.current_frame_index = 0 if num_frames > 0 else -1
        self.current_animation_enabled = num_frames > 0

        self.update_callbacks: List[Callable[[Property], None]] = []

    # Public API
    def get_current(self) -> PropertyItem:
        return self.get_frame_by_index(self.current_frame_index)

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

    def destroy_gpu_property(self) -> None:
        if self.gpu_property is not None:
            self.gpu_property.enqueue_for_destruction()
            self.gpu_property = None

    # Scene API
    def is_dynamic(self) -> bool:
        return self.animation.is_dynamic(self.num_frames)

    def update(self, time: float, frame: int) -> None:
        old_frame = self.current_frame_index
        self.current_frame_index = self.animation.get_frame_index(self.num_frames, time, frame)
        self.current_animation_enabled = not self.animation.is_disabled(self.num_frames, time, frame)

        if old_frame != self.current_frame_index:
            for c in self.update_callbacks:
                c(self)

    def start_animation_time(self, fps: float) -> float:
        return self.animation.start_animation_time(self.num_frames, fps)

    def end_animation_time(self, fps: float) -> float:
        return self.animation.end_animation_time(self.num_frames, fps)

    # Renderer API
    def create(self, r: "Renderer") -> None:
        pass

    def destroy(self) -> None:
        self.destroy_gpu_property()

    # Edit API
    def update_frame(self, frame_index: int, frame: Any) -> None:
        if self.gpu_property is not None:
            if self.gpu_property.supported_operations & gpu_property_mod.GpuPropertySupportedOperations.UPDATE:
                self.gpu_property.update_frame(frame_index)
            else:
                self.destroy_gpu_property()

    def update_frames(self, frame_indices: List[int], frames: Any) -> None:
        if self.gpu_property is not None:
            if self.gpu_property.supported_operations & gpu_property_mod.GpuPropertySupportedOperations.UPDATE:
                self.gpu_property.update_frames(frame_indices)
            else:
                self.destroy_gpu_property()

    def update_frame_range(self, start: int, frames: Any) -> None:
        if self.gpu_property is not None:
            if self.gpu_property.supported_operations & gpu_property_mod.GpuPropertySupportedOperations.UPDATE:
                self.gpu_property.update_frame_range(start, start + len(frames))
            else:
                self.destroy_gpu_property()

    def append_frame(self, frame: Any) -> None:
        if self.gpu_property is not None:
            if self.gpu_property.supported_operations & gpu_property_mod.GpuPropertySupportedOperations.APPEND:
                self.gpu_property.append_frame()
            else:
                self.destroy_gpu_property()

    def append_frames(self, frames: Any) -> None:
        if self.gpu_property is not None:
            if self.gpu_property.supported_operations & gpu_property_mod.GpuPropertySupportedOperations.APPEND:
                self.gpu_property.append_frames(len(frames))
            else:
                self.destroy_gpu_property()

    def pop_frame(self) -> None:
        if self.gpu_property is not None:
            if self.gpu_property.supported_operations & gpu_property_mod.GpuPropertySupportedOperations.POP:
                self.gpu_property.pop_frame()
            else:
                self.destroy_gpu_property()

    def pop_frames(self, count: int) -> None:
        if self.gpu_property is not None:
            if self.gpu_property.supported_operations & gpu_property_mod.GpuPropertySupportedOperations.POP:
                self.gpu_property.pop_frames(count)
            else:
                self.destroy_gpu_property()

    def insert_frame(self, frame_index: int, frame: Any) -> None:
        if self.gpu_property is not None:
            if self.gpu_property.supported_operations & gpu_property_mod.GpuPropertySupportedOperations.INSERT:
                self.gpu_property.insert_frame(frame_index)
            else:
                self.destroy_gpu_property()

    def insert_frames(self, index: int, frames: Any) -> None:
        if self.gpu_property is not None:
            if self.gpu_property.supported_operations & gpu_property_mod.GpuPropertySupportedOperations.INSERT:
                self.gpu_property.insert_frames(index, len(frames))
            else:
                self.destroy_gpu_property()

    def remove_frame(self, frame_index: int) -> None:
        if self.gpu_property is not None:
            if self.gpu_property.supported_operations & gpu_property_mod.GpuPropertySupportedOperations.REMOVE:
                self.gpu_property.remove_frame(frame_index)
            else:
                self.destroy_gpu_property()

    def remove_frames(self, frame_indices: List[int]) -> None:
        if self.gpu_property is not None:
            if self.gpu_property.supported_operations & gpu_property_mod.GpuPropertySupportedOperations.REMOVE:
                self.gpu_property.remove_frames(frame_indices)
            else:
                self.destroy_gpu_property()

    def remove_frame_range(self, start: int, stop: int) -> None:
        if self.gpu_property is not None:
            if self.gpu_property.supported_operations & gpu_property_mod.GpuPropertySupportedOperations.REMOVE:
                self.gpu_property.remove_frame_range(start, stop)
            else:
                self.destroy_gpu_property()


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

        self.gpu_property: Optional[gpu_property_mod.GpuProperty[gpu_property_mod.GpuBufferView]] = None
        self.gpu_usage = BufferUsageFlags(0)
        self.gpu_stage = PipelineStageFlags(0)

    # Public API
    def use_gpu(self, usage: BufferUsageFlags, stage: PipelineStageFlags) -> "BufferProperty":
        self.gpu_usage |= usage
        self.gpu_stage |= stage
        return self

    def get_current_gpu(self) -> gpu_property_mod.GpuBufferView:
        assert self.gpu_property is not None
        return self.gpu_property.get_current()

    # Renderer API
    def create(self, r: "Renderer") -> None:
        if self.gpu_usage and self.gpu_property is None:
            if self.upload.preupload:
                self.gpu_property = gpu_property_mod.GpuBufferPreuploadedProperty(r, self, self.name)
            else:
                self.gpu_property = gpu_property_mod.GpuBufferStreamingProperty(r, self, self.name)


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

        self.gpu_property: Optional[gpu_property_mod.GpuProperty[gpu_property_mod.GpuImageView]] = None
        self.gpu_usage = ImageUsageFlags(0)
        self.gpu_stage = PipelineStageFlags(0)
        self.gpu_layout = ImageLayout.UNDEFINED
        self.gpu_srgb = False
        self.gpu_mips = False

    # Public API
    def use_gpu(
        self,
        usage: ImageUsageFlags,
        layout: ImageLayout,
        stage: PipelineStageFlags,
        srgb: bool = False,
        mips: bool = False,
    ) -> "ImageProperty":
        if self.gpu_layout == ImageLayout.UNDEFINED:
            self.gpu_layout = layout
        elif self.gpu_layout != layout:
            # Fallback to general if we need multiple layouts
            self.gpu_layout = ImageLayout.GENERAL
        self.gpu_usage |= usage
        self.gpu_stage |= stage
        self.gpu_srgb |= srgb
        self.gpu_mips |= mips
        return self

    def get_current_gpu(self) -> gpu_property_mod.GpuImageView:
        assert self.gpu_property is not None
        return self.gpu_property.get_current()

    # Renderer API
    def create(self, r: "Renderer") -> None:
        if self.gpu_usage and self.gpu_property is None:
            if self.upload.preupload:
                self.gpu_property = gpu_property_mod.GpuImagePreuploadedProperty(r, self, self.name)
            else:
                self.gpu_property = gpu_property_mod.GpuImageStreamingProperty(r, self, self.name)


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
        max_size = data.itemsize * np.prod(data.shape[1:], dtype=np.int64)
        super().__init__(int(max_size), len(data), data.dtype, data.shape[1:], animation, upload, name)
        self.data = data

    # Public API
    def get_frame_by_index(self, frame_index: int, thread_index: int = -1) -> PropertyItem:
        return self.data[frame_index]  # type: ignore

    # Renderer API
    def create(self, r: "Renderer") -> None:
        if self.gpu_usage and self.gpu_property is None:
            if self.upload.preupload:
                if self.upload.batched:
                    self.gpu_property = gpu_property_mod.GpuBufferPreuploadedArrayProperty(
                        self.data, r, self, self.name
                    )
                else:
                    self.gpu_property = gpu_property_mod.GpuBufferPreuploadedProperty(r, self, self.name)
            else:
                self.gpu_property = gpu_property_mod.GpuBufferStreamingProperty(r, self, self.name)

    # Edit API
    def update_frame(self, frame_index: int, frame: ArrayLike) -> None:
        self.data[frame_index] = frame
        super().update_frame(frame_index, frame)

    def update_frames(self, frame_indices: List[int], frames: ArrayLike) -> None:
        self.data[frame_indices] = frames
        super().update_frames(frame_indices, frames)

    def update_frame_range(self, start: int, frames: NDArray[Any]) -> None:
        self.data[start : start + len(frames)] = frames
        super().update_frame_range(start, frames)

    def append_frame(self, frame: ArrayLike) -> None:
        self.data = np.append(self.data, np.asarray(frame)[np.newaxis], axis=0)
        self.num_frames = self.data.shape[0]
        super().append_frame(frame)

    def append_frames(self, frames: ArrayLike) -> None:
        self.data = np.append(self.data, frames, axis=0)
        self.num_frames = self.data.shape[0]
        super().append_frames(frames)

    def pop_frame(self) -> None:
        self.remove_frame(self.num_frames - 1)

    def pop_frames(self, count: int) -> None:
        self.remove_frame_range(self.num_frames - count, self.num_frames)

    def insert_frame(self, frame_index: int, frame: ArrayLike) -> None:
        self.data = np.insert(self.data, frame_index, frame, axis=0)
        self.num_frames = self.data.shape[0]
        super().insert_frame(frame_index, frame)

    def insert_frames(self, index: int, frames: ArrayLike) -> None:
        self.data = np.insert(self.data, index, frames, axis=0)
        self.num_frames = self.data.shape[0]
        super().insert_frames(index, frames)

    def remove_frame(self, frame_index: int) -> None:
        self.data = np.delete(self.data, frame_index, axis=0)
        self.num_frames = self.data.shape[0]
        super().remove_frame(frame_index)

    def remove_frames(self, frame_indices: List[int]) -> None:
        self.data = np.delete(self.data, frame_indices, axis=0)
        self.num_frames = self.data.shape[0]
        super().remove_frames(frame_indices)

    def remove_frame_range(self, start: int, stop: int) -> None:
        self.data = np.delete(self.data, slice(start, stop), axis=0)
        self.num_frames = self.data.shape[0]
        super().remove_frame_range(start, stop)


class ListBufferProperty(BufferProperty):
    def __init__(
        self,
        data: List[ArrayLike],
        dtype: Optional[DTypeLike] = None,
        shape: Tuple[int, ...] = (),
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

    # Public API
    def get_frame_by_index(self, frame_index: int, thread_index: int = -1) -> PropertyItem:
        return self.data[frame_index]

    # Edit API
    def update_frame(self, frame_index: int, frame: NDArray[Any]) -> None:
        self.data[frame_index] = frame
        super().update_frame(frame_index, frame)

    def update_frames(self, frame_indices: List[int], frames: List[NDArray[Any]]) -> None:
        for i, f in zip(frame_indices, frames):
            self.data[i] = f
        super().update_frames(frame_indices, frames)

    def update_frame_range(self, start: int, frames: List[NDArray[Any]]) -> None:
        self.data[start : len(frames)] = frames
        super().update_frame_range(start, frames)

    def append_frame(self, frame: NDArray[Any]) -> None:
        self.data.append(frame)
        self.num_frames = len(self.data)
        super().append_frame(frame)

    def append_frames(self, frames: List[NDArray[Any]]) -> None:
        self.data.extend(frames)
        self.num_frames = len(self.data)
        super().append_frames(frames)

    def pop_frame(self) -> None:
        self.data.pop()
        self.num_frames = len(self.data)
        super().pop_frame()

    def pop_frames(self, count: int) -> None:
        if count > 0:
            del self.data[-count:]
        self.num_frames = len(self.data)
        super().pop_frames(count)

    def insert_frame(self, frame_index: int, frame: NDArray[Any]) -> None:
        self.data.insert(frame_index, frame)
        self.num_frames = len(self.data)
        super().insert_frame(frame_index, frame)

    def insert_frames(self, index: int, frames: List[NDArray[Any]]) -> None:
        self.data = self.data[:index] + frames + self.data[index:]
        self.num_frames = len(self.data)
        super().insert_frames(index, frames)

    def remove_frame(self, frame_index: int) -> None:
        self.data.pop(frame_index)
        self.num_frames = len(self.data)
        super().remove_frame(frame_index)

    def remove_frames(self, frame_indices: List[int]) -> None:
        frame_indices_set = set(frame_indices)
        self.data = [d for i, d in enumerate(self.data) if i not in frame_indices_set]
        self.num_frames = len(self.data)
        super().remove_frames(frame_indices)

    def remove_frame_range(self, start: int, stop: int) -> None:
        self.data = self.data[:start] + self.data[stop:]
        self.num_frames = len(self.data)
        super().remove_frame_range(start, stop)


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

    # Public API
    def get_frame_by_index(self, frame_index: int, thread_index: int = -1) -> PropertyItem:
        return self.data[frame_index]  # type: ignore

    # Edit API
    def update_frame(self, frame_index: int, frame: ArrayLike) -> None:
        self.data[frame_index] = frame
        super().update_frame(frame_index, frame)

    def update_frames(self, frame_indices: List[int], frames: ArrayLike) -> None:
        self.data[frame_indices] = frames
        super().update_frames(frame_indices, frames)

    def update_frame_range(self, start: int, frames: NDArray[Any]) -> None:
        self.data[start : start + len(frames)] = frames
        super().update_frame_range(start, frames)

    def append_frame(self, frame: ArrayLike) -> None:
        self.data = np.append(self.data, np.asarray(frame)[np.newaxis], axis=0)
        self.num_frames = self.data.shape[0]
        super().append_frame(frame)

    def append_frames(self, frames: ArrayLike) -> None:
        self.data = np.append(self.data, frames, axis=0)
        self.num_frames = self.data.shape[0]
        super().append_frames(frames)

    def pop_frame(self) -> None:
        self.remove_frame(self.num_frames - 1)

    def pop_frames(self, count: int) -> None:
        self.remove_frame_range(self.num_frames - count, self.num_frames)

    def insert_frame(self, frame_index: int, frame: ArrayLike) -> None:
        self.data = np.insert(self.data, frame_index, frame, axis=0)
        self.num_frames = self.data.shape[0]
        super().insert_frame(frame_index, frame)

    def insert_frames(self, index: int, frames: ArrayLike) -> None:
        self.data = np.insert(self.data, index, frames, axis=0)
        self.num_frames = self.data.shape[0]
        super().insert_frames(index, frames)

    def remove_frame(self, frame_index: int) -> None:
        self.data = np.delete(self.data, frame_index, axis=0)
        self.num_frames = self.data.shape[0]
        super().remove_frame(frame_index)

    def remove_frames(self, frame_indices: List[int]) -> None:
        self.data = np.delete(self.data, frame_indices, axis=0)
        self.num_frames = self.data.shape[0]
        super().remove_frames(frame_indices)

    def remove_frame_range(self, start: int, stop: int) -> None:
        self.data = np.delete(self.data, slice(start, stop), axis=0)
        self.num_frames = self.data.shape[0]
        super().remove_frame_range(start, stop)


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
                    raise RuntimeError(
                        f"ListImageProperty data elements must all have the same shape. First has {property_shape}. Element {i} has {a.shape}"
                    )
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

    # Public API
    def get_frame_by_index(self, frame_index: int, thread_index: int = -1) -> PropertyItem:
        return self.data[frame_index]

    # Edit API
    def update_frame(self, frame_index: int, frame: NDArray[Any]) -> None:
        self.data[frame_index] = frame
        super().update_frame(frame_index, frame)

    def update_frames(self, frame_indices: List[int], frames: List[NDArray[Any]]) -> None:
        for i, f in zip(frame_indices, frames):
            self.data[i] = f
        super().update_frames(frame_indices, frames)

    def update_frame_range(self, start: int, frames: List[NDArray[Any]]) -> None:
        self.data[start : len(frames)] = frames
        super().update_frame_range(start, frames)

    def append_frame(self, frame: NDArray[Any]) -> None:
        self.data.append(frame)
        self.num_frames = len(self.data)
        super().append_frame(frame)

    def append_frames(self, frames: List[NDArray[Any]]) -> None:
        self.data.extend(frames)
        self.num_frames = len(self.data)
        super().append_frames(frames)

    def pop_frame(self) -> None:
        self.data.pop()
        self.num_frames = len(self.data)
        super().pop_frame()

    def pop_frames(self, count: int) -> None:
        if count > 0:
            del self.data[-count:]
        self.num_frames = len(self.data)
        super().pop_frames(count)

    def insert_frame(self, frame_index: int, frame: NDArray[Any]) -> None:
        self.data.insert(frame_index, frame)
        self.num_frames = len(self.data)
        super().insert_frame(frame_index, frame)

    def insert_frames(self, index: int, frames: List[NDArray[Any]]) -> None:
        self.data = self.data[:index] + frames + self.data[index:]
        self.num_frames = len(self.data)
        super().insert_frames(index, frames)

    def remove_frame(self, frame_index: int) -> None:
        self.data.pop(frame_index)
        self.num_frames = len(self.data)
        super().remove_frame(frame_index)

    def remove_frames(self, frame_indices: List[int]) -> None:
        frame_indices_set = set(frame_indices)
        self.data = [d for i, d in enumerate(self.data) if i not in frame_indices_set]
        self.num_frames = len(self.data)
        super().remove_frames(frame_indices)

    def remove_frame_range(self, start: int, stop: int) -> None:
        self.data = self.data[:start] + self.data[stop:]
        self.num_frames = len(self.data)
        super().remove_frame_range(start, stop)


def as_buffer_property(
    value: Union[BufferProperty, List[ArrayLike], ArrayLike],
    dtype: Optional[DTypeLike] = None,
    shape: Tuple[int, ...] = (),
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
        return ListBufferProperty(value, dtype, shape, name=name)
    else:
        value = np.atleast_1d(np.asarray(value, dtype, order="C"))
        if shape is None:
            if len(value.shape) > 1:
                raise ShapeError((1,), value.shape[1:])
        else:
            if shape_match(shape, value.shape):
                # Implicitly add animation dimension
                value = value[np.newaxis]
            elif shape_match(shape, value.shape[1:]):
                # Shape already matches
                pass
            else:
                raise ShapeError(shape, value.shape)
        return ArrayBufferProperty(value, dtype, name=name)


def as_image_property(
    value: Union[ImageProperty, List[ArrayLike], ArrayLike],
    name: str = "",
) -> ImageProperty:
    if isinstance(value, ImageProperty):
        if not value.name:
            value.name = name
        return value
    elif isinstance(value, List):
        return ListImageProperty(value, None, name=name)
    else:
        data = np.asarray(value)
        if data.ndim == 3:
            # Implicitly add animation dimension
            data = data[np.newaxis]
        elif data.ndim == 4:
            # Shape already matches
            pass
        else:
            raise ShapeError((-1, -1, -1), data.shape)

        return ArrayImageProperty(data, None, name=name)
