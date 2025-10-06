# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from typing import Callable, List, Optional, Set, Tuple, Union

import numpy as np
from numpy.typing import DTypeLike
from pyglm.glm import mat3, mat4, quat, vec2, vec3
from pyxpg import DescriptorSet, imgui

from . import lights, materials, renderer
from .property import (
    Animation,
    BufferProperty,
    ImageProperty,
    Property,
    PropertyData,
    UploadSettings,
    as_buffer_property,
    as_image_property,
)
from .renderer_frame import RendererFrame
from .transform2d import Transform2D
from .transform3d import Transform3D

_counter = 0


class Object:
    def __init__(
        self,
        name: Optional[str] = None,
        material: Optional["materials.Material"] = None,
    ):
        self.uid = Object.next_id()
        self.name = name or f"{type(self).__name__}<{self.uid}>"
        self.children: List[Object] = []
        self.properties: List[Property] = []
        self.material = material

        self.created = False
        self.gui_expanded = False
        self.gui_selected = False
        self.gui_selected_property: Optional[Property] = None

    @staticmethod
    def next_id() -> int:
        global _counter
        _counter += 1
        return _counter

    def add_buffer_property(
        self,
        prop: Union[BufferProperty, PropertyData, int, float],
        dtype: Optional[DTypeLike] = None,
        shape: Optional[Tuple[int, ...]] = None,
        animation: Optional[Animation] = None,
        upload: Optional[UploadSettings] = None,
        name: str = "",
    ) -> BufferProperty:
        property = as_buffer_property(prop, dtype, shape, animation, upload, name)
        self.properties.append(property)
        return property

    def add_image_property(
        self,
        prop: Union[ImageProperty, PropertyData],
        animation: Optional[Animation] = None,
        upload: Optional[UploadSettings] = None,
        name: str = "",
    ) -> ImageProperty:
        property = as_image_property(prop, animation, upload, name)
        self.properties.append(property)
        return property

    def create_if_needed(self, renderer: "renderer.Renderer") -> None:
        if self.material is not None:
            self.material.create_if_needed(renderer)

        if not self.created:
            self.create(renderer)
            self.created = True

    def create(self, renderer: "renderer.Renderer") -> None:
        pass

    def collect_dynamic_properties(self, all_properties: Set[Property]) -> None:
        for p in self.properties:
            if p.num_frames > 1:
                all_properties.add(p)
        if self.material is not None:
            for mp in self.material.properties:
                if mp.property.num_frames > 1:
                    all_properties.add(mp.property)

    def update(self, time: float, frame: int) -> None:
        pass

    def update_transform(self, parent: Optional["Object"]) -> None:
        # TODO: can merge with udpate? Where to find parent? With link?
        pass

    def upload(self, renderer: "renderer.Renderer", frame: RendererFrame) -> None:
        pass

    def render(self, renderer: "renderer.Renderer", frame: RendererFrame, scene_descriptor_set: DescriptorSet) -> None:
        pass

    def render_depth(
        self, renderer: "renderer.Renderer", frame: RendererFrame, scene_descriptor_set: DescriptorSet
    ) -> None:
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

                if isinstance(p, BufferProperty):
                    imgui.text(f"{p.shape} {np.dtype(p.dtype).name} {p.max_size}")
                elif isinstance(p, ImageProperty):
                    imgui.text(f"[{p.width}x{p.height}] {p.format}")

                if p.num_frames > 1:
                    imgui.text(f"{p.current_frame_index} / {p.num_frames}")
                imgui.indent(-10)
        imgui.indent(-5)


class Object2D(Object):
    def __init__(
        self,
        name: Optional[str],
        translation: Optional[BufferProperty] = None,
        rotation: Optional[BufferProperty] = None,
        scale: Optional[BufferProperty] = None,
    ):
        super().__init__(name)
        self.translation = self.add_buffer_property(
            translation if translation is not None else np.array([0, 0]),
            np.float32,
            (2,),
            name="translation",
        )
        self.rotation = self.add_buffer_property(
            rotation if rotation is not None else np.array([0]),
            np.float32,
            name="rotation",
        )
        self.scale = self.add_buffer_property(
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
        translation: Optional[BufferProperty] = None,
        rotation: Optional[BufferProperty] = None,
        scale: Optional[BufferProperty] = None,
        material: Optional["materials.Material"] = None,
    ):
        super().__init__(name, material)
        self.translation = self.add_buffer_property(
            translation if translation is not None else np.array([0, 0, 0]),
            np.float32,
            (3,),
            name="translation",
        )
        self.rotation = self.add_buffer_property(
            rotation if rotation is not None else np.array([1, 0, 0, 0]),
            np.float32,
            (4,),
            name="rotation",
        )
        self.scale = self.add_buffer_property(
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
            parent.current_transform_matrix if parent is not None else mat4(1.0)
        ) * self.current_relative_transform.as_mat4()  # type: ignore


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

        all_dynamic_properties: Set[Property] = set()

        def visit(o: Object) -> None:
            o.collect_dynamic_properties(all_dynamic_properties)

        self.visit_objects(visit)

        for p in all_dynamic_properties:
            time = max(time, p.max_animation_time(frames_per_second))

        return time

    def update(self, time: float, frame: int) -> None:
        all_dynamic_properties: Set[Property] = set()

        def collect(o: Object) -> None:
            o.collect_dynamic_properties(all_dynamic_properties)

        self.visit_objects(collect)

        for p in all_dynamic_properties:
            p.update(time, frame)

        def update(p: Optional[Object], o: Object) -> None:
            o.update_transform(p)

        self.visit_objects_with_parent(update)

    def create_if_needed(self, renderer: "renderer.Renderer") -> None:
        def visit(o: Object) -> None:
            o.create_if_needed(renderer)

        self.visit_objects(visit)

    def render(self, renderer: "renderer.Renderer", frame: RendererFrame, descriptor_set: DescriptorSet) -> None:
        def visit(o: Object) -> None:
            o.render(renderer, frame, descriptor_set)

        self.visit_objects(visit)

    def upload(self, renderer: "renderer.Renderer", frame: RendererFrame) -> None:
        def visit(o: Object) -> None:
            if o.material is not None:
                o.material.upload(renderer, frame)
            o.upload(renderer, frame)

        self.visit_objects(visit)

    def render_depth(self, renderer: "renderer.Renderer", frame: RendererFrame, descriptor_set: DescriptorSet) -> None:
        def visit(o: Object) -> None:
            o.render_depth(renderer, frame, descriptor_set)

        self.visit_objects(visit)

    def render_shadowmaps(self, renderer: "renderer.Renderer", frame: RendererFrame) -> None:
        def visit(o: Object) -> None:
            if isinstance(o, lights.Light):
                o.render_shadowmaps(renderer, frame, self)

        self.visit_objects(visit)
