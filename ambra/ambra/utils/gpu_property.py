from pyxpg import BufferUsageFlags, ImageUsageFlags, Format, ImageUsage

from ..scene import Object, Property
from ..renderer import Renderer
from .gpu import UploadableBuffer, UploadableImage
from typing import List

import numpy as np

def view_bytes(a: np.ndarray) -> memoryview:
    return a.reshape((-1,), copy=False).view(np.uint8).data

class GpuBufferProperty:
    def __init__(self, o: Object, r: Renderer, property: Property[np.ndarray], usage_flags: BufferUsageFlags, name: str = None):
        self.property = property

        # Upload
        prefer_preupload = r.prefer_preupload if property.prefer_preupload is None else property.prefer_preupload
        if prefer_preupload:
            self.buffers = [
                UploadableBuffer.from_data(r.ctx, view_bytes(property.get_frame_by_index(i)), usage_flags, name)
                for i in range(property.num_frames)
            ]
        else:
            raise NotImplemented()

        o.update_callbacks.append(lambda time, frame: self.update(time, frame))
        o.destroy_callbacks.append(lambda: self.destroy())

    def update(self, time: int, frame: float):
        pass

    def get_current(self):
        return self.buffers[self.property.current_frame]

    def destroy(self):
        for buf in self.buffers:
            buf.destroy()
        self.buffers.clear()

_channels_dtype_int_to_format_table = {
    # normalized formats
    (1, np.dtype(np.uint8),  False): Format.R8_UNORM,
    (2, np.dtype(np.uint8),  False): Format.R8G8_UNORM,
    (3, np.dtype(np.uint8),  False): Format.R8G8B8_UNORM,
    (4, np.dtype(np.uint8),  False): Format.R8G8B8A8_UNORM,
    (1, np.dtype(np.int8),   False): Format.R8_SNORM,
    (2, np.dtype(np.int8),   False): Format.R8G8_SNORM,
    (3, np.dtype(np.int8),   False): Format.R8G8B8_SNORM,
    (4, np.dtype(np.int8),   False): Format.R8G8B8A8_SNORM,
    (1, np.dtype(np.uint16), False): Format.R16_UNORM,
    (2, np.dtype(np.uint16), False): Format.R16G16_UNORM,
    (3, np.dtype(np.uint16), False): Format.R16G16B16_UNORM,
    (4, np.dtype(np.uint16), False): Format.R16G16B16A16_UNORM,
    (1, np.dtype(np.int16),  False): Format.R16_SNORM,
    (2, np.dtype(np.int16),  False): Format.R16G16_SNORM,
    (3, np.dtype(np.int16),  False): Format.R16G16B16_SNORM,
    (4, np.dtype(np.int16),  False): Format.R16G16B16A16_SNORM,

    # integer formats
    (1, np.dtype(np.uint8),  True): Format.R8_UINT,
    (2, np.dtype(np.uint8),  True): Format.R8G8_UINT,
    (3, np.dtype(np.uint8),  True): Format.R8G8B8_UINT,
    (4, np.dtype(np.uint8),  True): Format.R8G8B8A8_UINT,
    (1, np.dtype(np.int8),   True): Format.R8_SINT,
    (2, np.dtype(np.int8),   True): Format.R8G8_SINT,
    (3, np.dtype(np.int8),   True): Format.R8G8B8_SINT,
    (4, np.dtype(np.int8),   True): Format.R8G8B8A8_SINT,
    (1, np.dtype(np.uint16), True): Format.R16_UINT,
    (2, np.dtype(np.uint16), True): Format.R16G16_UINT,
    (3, np.dtype(np.uint16), True): Format.R16G16B16_UINT,
    (4, np.dtype(np.uint16), True): Format.R16G16B16A16_UINT,
    (1, np.dtype(np.int16),  True): Format.R16_SINT,
    (2, np.dtype(np.int16),  True): Format.R16G16_SINT,
    (3, np.dtype(np.int16),  True): Format.R16G16B16_SINT,
    (4, np.dtype(np.int16),  True): Format.R16G16B16A16_SINT,
    (1, np.dtype(np.uint32), True): Format.R32_UINT,
    (2, np.dtype(np.uint32), True): Format.R32G32_UINT,
    (3, np.dtype(np.uint32), True): Format.R32G32B32_UINT,
    (4, np.dtype(np.uint32), True): Format.R32G32B32A32_UINT,
    (1, np.dtype(np.int32),  True): Format.R32_SINT,
    (2, np.dtype(np.int32),  True): Format.R32G32_SINT,
    (3, np.dtype(np.int32),  True): Format.R32G32B32_SINT,
    (4, np.dtype(np.int32),  True): Format.R32G32B32A32_SINT,

    # float formats
    (1, np.dtype(np.float16), False): Format.R16_SFLOAT,
    (2, np.dtype(np.float16), False): Format.R16G16_SFLOAT,
    (3, np.dtype(np.float16), False): Format.R16G16B16_SFLOAT,
    (4, np.dtype(np.float16), False): Format.R16G16B16A16_SFLOAT,
    (1, np.dtype(np.float32), False): Format.R32_SFLOAT,
    (2, np.dtype(np.float32), False): Format.R32G32_SFLOAT,
    (3, np.dtype(np.float32), False): Format.R32G32B32_SFLOAT,
    (4, np.dtype(np.float32), False): Format.R32G32B32A32_SFLOAT,
    (1, np.dtype(np.float64), False): Format.R64_SFLOAT,
    (2, np.dtype(np.float64), False): Format.R64G64_SFLOAT,
    (3, np.dtype(np.float64), False): Format.R64G64B64_SFLOAT,
    (4, np.dtype(np.float64), False): Format.R64G64B64A64_SFLOAT,
}

class GpuImageProperty:
    def __init__(self, o: Object, r: Renderer, property: Property[np.ndarray], usage_flags: ImageUsageFlags, usage: ImageUsage, name: str = None):
        self.property = property

        # Upload
        prefer_preupload = r.prefer_preupload if property.prefer_preupload is None else property.prefer_preupload
        if prefer_preupload:
            self.images: List[UploadableImage] = []
            for i in range(property.num_frames):
                frame = property.get_frame_by_index(i)
                if len(frame.shape) != 3:
                    raise ValueError(f"Expected shape of length 3. Got: {len(frame.shape)}")

                height, width, channels = frame.shape
                try:
                    format = _channels_dtype_int_to_format_table[(channels, frame.dtype, False)]
                except KeyError:
                    raise ValueError(f"Combination of channels ({channels}) and dtype ({frame.dtype}) does not match any supported image format")

                img = UploadableImage.from_data(r.ctx, view_bytes(frame), usage, width, height, format, usage_flags, name)
                self.images.append(img)
        else:
            raise NotImplemented()

        o.update_callbacks.append(lambda time, frame: self.update(time, frame))
        o.destroy_callbacks.append(lambda: self.destroy())

    def update(self, time: int, frame: float):
        pass

    def get_current(self):
        return self.images[self.property.current_frame]

    def destroy(self):
        for img in self.images:
            img.destroy()
        self.images.clear()

