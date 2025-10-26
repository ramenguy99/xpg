from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from pyglm.glm import inverse, mat3, mat4x3, transpose
from pyxpg import (
    Attachment,
    BlendFactor,
    BlendOp,
    BufferUsageFlags,
    CullMode,
    Depth,
    DescriptorSet,
    DeviceFeatures,
    Format,
    FrontFace,
    GraphicsPipeline,
    InputAssembly,
    MemoryUsage,
    PipelineStage,
    PipelineStageFlags,
    PrimitiveTopology,
    PushConstantsRange,
    Rasterization,
    Shader,
    Stage,
    VertexAttribute,
    VertexBinding,
    VertexInputRate,
)

from ambra.property import BufferProperty, ImageProperty, as_image_property
from ambra.renderer import Renderer
from ambra.renderer_frame import RendererFrame
from ambra.scene import Object, Object3D
from ambra.utils.gpu import cull_mode_opposite_face


@dataclass
class Splats:
    position: np.ndarray
    sh: np.ndarray
    opacity: np.ndarray
    scale: np.ndarray
    rotation: np.ndarray


def load_ply(path: Path, sh_degree: int = 1):
    with open(path, "rb") as f:
        # Read header.
        head = f.readline().decode("utf-8").strip().lower()
        if head != "ply":
            print(head)
            raise ValueError(f"Not a ply file: {head}")

        encoding = f.readline().decode("utf-8").strip().lower()
        if "binary_little_endian" not in encoding:
            raise ValueError(f"Invalid encoding: {encoding}")

        elements = f.readline().decode("utf-8").strip().lower()
        count = int(elements.split()[2])

        # Read until end of header.
        while f.readline().decode("utf-8").strip().lower() != "end_header":
            pass

        # Number of 32 bit floats used to encode Spherical Harmonics coefficients.
        # The last multiplication by 3 is because we have 3 components (RGB) for each coefficient.
        sh_coeffs = (sh_degree + 1) * (sh_degree + 1) * 3

        # Position (vec3), normal (vec3), spherical harmonics (sh_coeffs), opacity (float),
        # scale (vec3) and rotation (quaternion). All values are float32 (4 bytes).
        size = count * (3 + 3 + sh_coeffs + 1 + 3 + 4) * 4

        data = f.read(size)
        arr = np.frombuffer(data, dtype=np.float32).reshape((count, -1))

        # Positions.
        position = arr[:, :3].copy()

        # Currently we don't need normals for rendering.
        # normal = arr[:, 3:6].copy()

        # Spherical harmonic coefficients.
        sh = arr[:, 6 : 6 + sh_coeffs].copy()

        # Activate alpha: sigmoid(alpha).
        opacity = 1.0 / (1.0 + np.exp(-arr[:, 6 + sh_coeffs]))

        # Exponentiate scale.
        scale = np.exp(arr[:, 7 + sh_coeffs : 10 + sh_coeffs])

        # Normalize quaternions.
        rotation = arr[:, 10 + sh_coeffs : 14 + sh_coeffs].copy()
        rotation /= np.linalg.norm(rotation, ord=2, axis=1)[..., np.newaxis]

        # Convert from wxyz to xyzw.
        rotation = np.roll(rotation, -1, axis=1)

        return Splats(position, sh, opacity, scale, rotation)


class GaussianSplats(Object3D):
    def __init__(
        self,
        splats: Splats,
        name: Optional[str] = None,
        translation: Optional[BufferProperty] = None,
        rotation: Optional[BufferProperty] = None,
        scale: Optional[BufferProperty] = None,
    ):
        super().__init__(name, translation, rotation, scale)

    def create(self, r: Renderer) -> None:
        distance = r.compile_builtin_shader("/basic.slang", "vertex_main")
        sort = r.compile_builtin_shader("3d/basic.slang", "vertex_main")

        vert = r.compile_builtin_shader("3d/basic.slang", "vertex_main")
        frag = r.compile_builtin_shader("3d/basic.slang", "pixel_main")

        # Instantiate the pipeline using the compiled shaders
        self.pipeline = GraphicsPipeline(
            r.ctx,
            stages=[
                PipelineStage(Shader(r.ctx, vert.code), Stage.VERTEX),
                PipelineStage(Shader(r.ctx, frag.code), Stage.FRAGMENT),
            ],
            vertex_bindings=[
                VertexBinding(0, 12, VertexInputRate.VERTEX),
                VertexBinding(1, 4, VertexInputRate.VERTEX),
            ],
            vertex_attributes=[
                VertexAttribute(0, 0, Format.R32G32B32_SFLOAT),
                VertexAttribute(1, 1, Format.R32_UINT),
            ],
            rasterization=Rasterization(dynamic_line_width=True),
            input_assembly=InputAssembly(
                PrimitiveTopology.LINE_STRIP if self.is_strip else PrimitiveTopology.LINE_LIST
            ),
            samples=r.msaa_samples,
            attachments=[Attachment(format=r.output_format)],
            depth=Depth(r.depth_format, True, True, r.depth_compare_op),
            descriptor_set_layouts=[
                r.scene_descriptor_set_layout,
            ],
            push_constants_ranges=[PushConstantsRange(self.constants_dtype.itemsize)],
        )

    def update_transform(self, parent: Optional[Object]) -> None:
        super().update_transform(parent)
        self.constants["transform"] = mat4x3(self.current_transform_matrix)

    def render(self, r: Renderer, frame: RendererFrame, scene_descriptor_set: DescriptorSet) -> None:
        frame.cmd.bind_graphics_pipeline(
            self.pipeline,
            vertex_buffers=[
                self.lines_buffer.get_current(),
                self.colors_buffer.get_current(),
            ],
            descriptor_sets=[
                scene_descriptor_set,
            ],
            push_constants=self.constants.tobytes(),
        )
        frame.cmd.set_line_width(
            self.line_width.get_current().item() if r.ctx.device_features & DeviceFeatures.WIDE_LINES else 1.0
        )

        frame.cmd.draw(self.lines.get_current().shape[0])
