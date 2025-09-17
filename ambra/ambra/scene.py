from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import DTypeLike
from pyglm.glm import mat3, mat4, quat, vec2, vec3
from pyxpg import imgui

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
from .transform2d import Transform2D
from .transform3d import Transform3D

_counter = 0


class Object:
    def __init__(self, name: Optional[str] = None):
        self.uid = Object.next_id()
        self.name = name or f"{type(self).__name__}<{self.uid}>"
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

    def upload(self, renderer, frame):  # type: ignore
        pass

    def render(self, renderer, frame, scene_descriptor_set):  # type: ignore
        pass

    def render_depth(self, renderer, frame, scene_descriptor_set):  # type: ignore
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
    ):
        super().__init__(name)
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


class Light(Object3D):
    def render_shadowmaps(self, renderer, frame, scene: "Scene") -> None:
        pass


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

    def render(self, renderer, frame, descriptor_set) -> None:  # type: ignore
        def visit(o: Object) -> None:
            o.render(renderer, frame, descriptor_set)  # type: ignore

        self.visit_objects(visit)

    def upload(self, renderer, frame) -> None:  # type: ignore
        def visit(o: Object) -> None:
            o.upload(renderer, frame)  # type: ignore

        self.visit_objects(visit)

    def render_depth(self, renderer, frame, descriptor_set) -> None:  # type: ignore
        def visit(o: Object) -> None:
            o.render_depth(renderer, frame, descriptor_set)  # type: ignore

        self.visit_objects(visit)

    def render_shadowmaps(self, renderer, frame) -> None:  # type: ignore
        def visit(o: Object) -> None:
            if isinstance(o, Light):
                o.render_shadowmaps(renderer, frame, self)  # type: ignore

        self.visit_objects(visit)
