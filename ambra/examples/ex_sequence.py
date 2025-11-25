import logging
import struct
import sys

import numpy as np
from pyglm.glm import ivec2, normalize, quatLookAtRH, vec3
from pyxpg import *

from ambra.config import CameraConfig, Config, GuiConfig, PlaybackConfig, RendererConfig, UploadMethod
from ambra.lights import DirectionalLight, DirectionalShadowSettings
from ambra.primitives3d import Mesh
from ambra.property import BufferProperty, UploadSettings
from ambra.utils.descriptors import create_descriptor_layout_pool_and_set
from ambra.utils.hook import hook
from ambra.utils.io import (
    read_exact,
    read_exact_at_offset,
    read_exact_at_offset_into,
)
from ambra.viewer import Action, Key, Modifiers, Viewer, imgui

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s.%(msecs)03d] %(levelname)-6s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

NUM_WORKERS = 4

files = [open(sys.argv[1], "rb", buffering=0) for _ in range(NUM_WORKERS)]
file = files[0]
header = read_exact(file, 12)
N = struct.unpack("<I", header[0:4])[0]
V = struct.unpack("<I", header[4:8])[0]
I = struct.unpack("<I", header[8:12])[0]

indices = np.frombuffer(read_exact_at_offset(file, N * V * 12 + len(header), I * 4), np.uint32)

scale = 1.0


class FileStreamingProperty(BufferProperty):
    def _get_size_offset(frame_index: int):
        return V * 12, 12 + frame_index * V * 12

    # def get_frame_by_index_into(self, frame_index: int, out: memoryview, thread_index: int = -1) -> int:
    #     size, offset = FileStreamingProperty._get_size_offset(frame_index)
    #     read_exact_at_offset_into(files[thread_index], offset, out[:size])
    #     return size

    def get_frame_by_index(self, frame_index: int, thread_index: int = -1):
        size, offset = FileStreamingProperty._get_size_offset(frame_index)
        buf = np.empty(size, np.uint8)
        read_exact_at_offset_into(files[thread_index], offset, buf.data)
        return buf.view(np.float32).reshape((-1, 3)) * scale


positions = FileStreamingProperty(
    V * 12,
    N,
    np.float32,
    (-1, 3),
    upload=UploadSettings(
        preupload=False,
        async_load=True,
        cpu_prefetch_count=2,
        gpu_prefetch_count=2,
    ),
)

mesh = Mesh(positions, indices=indices)


class CustomViewer(Viewer):
    def __init__(self, title="ambra", config=None, key_map=None):
        super().__init__(title, config, key_map)
        self._texture = None

    @hook
    def on_gui(self):
        global scale
        if imgui.begin("Control"):
            u, scale = imgui.slider_float("SCALE", scale, 0.1, 10.0)
            if u:
                mesh.positions.update_frame(self.playback.current_frame, None)
        imgui.end()
        if imgui.begin("Image"):
            if hasattr(light, "shadow_map"):
                if self._texture is None and light.shadow_map is not None:
                    sampler = Sampler(
                        viewer.ctx,
                        u=SamplerAddressMode.REPEAT,
                        v=SamplerAddressMode.REPEAT,
                    )
                    layout, pool, set = create_descriptor_layout_pool_and_set(
                        viewer.ctx,
                        [
                            DescriptorSetBinding(1, DescriptorType.COMBINED_IMAGE_SAMPLER, stage_flags=Stage.FRAGMENT),
                        ],
                    )
                    set.write_combined_image_sampler(
                        light.shadow_map, ImageLayout.SHADER_READ_ONLY_OPTIMAL, sampler, 0
                    )
                    self._sampler = sampler
                    self._texture = imgui.Texture(set)

                if self._texture is not None:
                    avail = imgui.get_content_region_avail()
                    available = ivec2(avail.x, avail.y)

                    ar = 1.0

                    # height = available.x / ar
                    height = available.y
                    view_size = ivec2(ar * height, height)
                    imgui.image(
                        self._texture,
                        imgui.Vec2(*view_size),
                        uv1=(1, -1),
                    )
        imgui.end()

    def on_key(self, key: Key, action: Action, modifiers: Modifiers):
        handled = False
        if action == Action.PRESS or action == Action.REPEAT:
            if key == Key.A:
                mesh.positions.update_frame(self.playback.current_frame, None)
        if not handled:
            return super().on_key(key, action, modifiers)


viewer = CustomViewer(
    config=Config(
        window_x=10,
        window_y=50,
        window_width=1900,
        window_height=1000,
        # vsync = False,
        preferred_frames_in_flight=3,
        playback=PlaybackConfig(
            playing=True,
            frames_per_second=25.0,
        ),
        renderer=RendererConfig(
            background_color=(0, 0, 0, 1),
            # force_buffer_upload_method=UploadMethod.GRAPHICS_QUEUE,
            # force_buffer_upload_method=UploadMethod.TRANSFER_QUEUE,
            # force_buffer_upload_method=UploadMethod.MAPPED_PREFER_HOST,
            # force_buffer_upload_method=UploadMethod.MAPPED_PREFER_DEVICE,
            upload_buffer_count=2,
            thread_pool_workers=NUM_WORKERS,
        ),
        gui=GuiConfig(
            stats=True,
            playback=True,
            inspector=True,
            renderer=True,
        ),
        camera=CameraConfig(
            position=(10, -10, 10),
            target=(0, 0, 0),
        ),
        world_up=(0, -1, 0),
    ),
)

light_position = vec3(5, 5, 5)
light_target = vec3(0, 0, 0)

rotation = quatLookAtRH(normalize(light_target - light_position), vec3(0, 1, 0))
light = DirectionalLight(
    np.array([1.0, 1.0, 1.0]),
    shadow_settings=DirectionalShadowSettings(half_extent=5.0),
    translation=light_position,
    rotation=rotation,
)

viewer.scene.objects.append(mesh)
viewer.scene.objects.append(light)

viewer.run()
