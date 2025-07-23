from pyxpg import BufferUsageFlags

from ..scene import Object, Property
from ..renderer import Renderer
from .gpu import UploadableBuffer

import numpy as np

class GpuBufferProperty:
    def __init__(self, o: Object, r: Renderer, property: Property, usage_flags: BufferUsageFlags, name: str = None):
        self.property = property

        # Upload
        prefer_preupload = r.prefer_preupload if property.prefer_preupload is None else property.prefer_preupload
        if prefer_preupload:
            def view_bytes(a: np.ndarray) -> memoryview:
                return a.reshape((-1,), copy=False).view(np.uint8).data

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

