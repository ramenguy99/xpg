# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List, Optional

from numpy.typing import NDArray
from pyglm.glm import (
    acos,
    clamp,
    cross,
    distance,
    dot,
    dvec2,
    ivec2,
    mat3,
    mat3_cast,
    mat4,
    normalize,
    radians,
    rotate,
    sqrt,
    tan,
    translate,
    transpose,
    vec2,
    vec3,
    vec4,
)
from pyxpg import DescriptorSet, Image, ImageView, imgui

from .camera import Camera, CameraDepth, OrthographicCamera, PerspectiveCamera
from .config import CameraConfig, CameraControlMode, CameraProjection, Handedness, PlaybackConfig
from .grid import GridType
from .transform3d import RigidTransform3D
from .utils.gpu import UploadableBuffer
from .utils.ring_buffer import RingBuffer


class Playback:
    def __init__(self, config: PlaybackConfig):
        self.max_time = config.max_time
        self.num_frames = int(self.max_time * config.frames_per_second) if self.max_time is not None else 0

        self.playing = config.playing
        self.lock_to_last_frame = config.lock_to_last_frame

        self.frames_per_second = config.frames_per_second
        self.playback_speed_multiplier = config.playback_speed_multiplier

        if config.initial_frame is None:
            self.current_time = config.initial_time
            self.current_frame = int(self.current_time * self.frames_per_second)
        else:
            self.set_frame(config.initial_frame)

    def set_max_time(self, max_time: float) -> None:
        self.max_time = max(max_time, 0.0)
        self.num_frames = max(int(self.max_time * self.frames_per_second), 1)

    def step(self, dt: float) -> None:
        if self.lock_to_last_frame and self.max_time is not None:
            self.current_time = self.max_time
            self.current_frame = self.num_frames - 1
        else:
            self.set_time(self.current_time + dt * self.playback_speed_multiplier)

    def toggle_play_pause(self) -> None:
        self.playing = not self.playing

    def set_time(self, time: float) -> None:
        assert self.max_time is not None
        time = max(time, 0.0)
        self.current_time = math.fmod(time, self.max_time) if self.max_time != 0.0 else 0.0
        self.current_frame = int(self.current_time * self.frames_per_second)

    def set_frame(self, frame: int) -> None:
        self.current_frame = frame % self.num_frames
        self.current_time = self.current_frame / self.frames_per_second


@dataclass
class Rect:
    x: int
    y: int
    width: int
    height: int


@dataclass
class PathTracerViewport:
    descriptor_sets: RingBuffer[DescriptorSet]
    uniform_buffers: RingBuffer[UploadableBuffer]
    constants: NDArray[Any]
    sample_index: int
    last_camera_transform: RigidTransform3D
    accumulation_image: Optional[Image]


class OrthographicViewType(Enum):
    POSITIVE_X = auto()
    POSITIVE_Y = auto()
    POSITIVE_Z = auto()
    NEGATIVE_X = auto()
    NEGATIVE_Y = auto()
    NEGATIVE_Z = auto()


@dataclass
class ViewportImage:
    image: Image
    srgb_image_view: ImageView
    imgui_texture: imgui.Texture


class Viewport:
    def __init__(
        self,
        rect: Rect,
        playback: Playback,
        camera_config: CameraConfig,
        world_up: vec3,
        handedness: Handedness,
        scene_descriptor_sets: RingBuffer[DescriptorSet],
        scene_uniform_buffers: RingBuffer[UploadableBuffer],
        scene_light_buffers: RingBuffer[List[UploadableBuffer]],
        scene_constants: NDArray[Any],
        viewport_image: Optional[ViewportImage],
        name: str,
    ):
        camera_from_world = RigidTransform3D.look_at(
            vec3(camera_config.position),
            vec3(camera_config.target),
            world_up,
            handedness,
        )

        camera: Camera
        if camera_config.projection == CameraProjection.PERSPECTIVE:
            camera = PerspectiveCamera(
                camera_from_world,
                CameraDepth(camera_config.z_near, camera_config.z_far),
                camera_config.perspective_vertical_fov,
            )
        elif camera_config.projection == CameraProjection.ORTHOGRAPHIC:
            camera = OrthographicCamera(
                camera_from_world,
                CameraDepth(camera_config.z_near, camera_config.z_far),
                vec2(camera_config.ortho_center),
                vec2(camera_config.ortho_half_extents),
            )
        else:
            raise RuntimeError(f"Unhandled camera type {camera_config.projection}")

        self.rect = rect
        self.camera = camera
        self.playback = playback
        self.scene_descriptor_sets = scene_descriptor_sets
        self.scene_uniform_buffers = scene_uniform_buffers
        self.scene_light_buffers = scene_light_buffers
        self.scene_constants = scene_constants
        self.viewport_image = viewport_image
        self.name = name

        # Path tracer
        self.path_tracer_viewport: Optional[PathTracerViewport] = None

        # Config
        self.handedness = handedness
        self.camera_world_up = world_up
        self.camera_target = vec3(camera_config.target)
        self.camera_control_mode = camera_config.control_mode
        self.camera_rotation_speed = vec2(camera_config.rotation_speed)
        self.camera_pan_speed = vec2(camera_config.pan_speed)
        self.camera_pan_distance_speed_scale = camera_config.pan_distance_speed_scale
        self.camera_pan_min_speed_scale = camera_config.pan_min_speed_scale
        self.camera_zoom_speed = camera_config.zoom_speed
        self.camera_zoom_distance_speed_scale = camera_config.zoom_distance_speed_scale
        self.camera_zoom_min_speed_scale = camera_config.zoom_min_speed_scale
        self.camera_zoom_min_target_distance = camera_config.zoom_min_target_distance
        self.camera_zoom_ortho_speed = camera_config.zoom_ortho_speed

        # State
        self.rotate_pressed = False
        self.pan_pressed = False
        self.drag_start_mouse_position = ivec2(0, 0)
        self.drag_start_camera_inverse_view_rotation = mat3()
        self.drag_start_camera_position = vec3(0)
        self.drag_start_camera_target = vec3(0)
        self.drag_start_camera_right = vec3(0)
        self.drag_start_camera_up = vec3(0)
        self.drag_start_camera_pitch = 0.0
        self.is_temporary_ortho = False
        self.prev_camera_fov: Optional[float] = None
        self.prev_camera_depth: Optional[CameraDepth] = None
        self.prev_camera_world_up: Optional[vec3] = None
        self.temporary_ortho_grid_type: GridType = GridType.XY_PLANE

    def resize(self, width: int, height: int) -> None:
        if self.path_tracer_viewport is not None and (self.rect.width != width or self.rect.height != height):
            # Invalidate accumulation after resize
            self.path_tracer_viewport.sample_index = 0
        self.rect.width = width
        self.rect.height = height

    def start_drag(self, position: ivec2) -> None:
        self.drag_start_mouse_position = position
        self.drag_start_camera_inverse_view_rotation = transpose(mat3_cast(self.camera.camera_from_world.rotation))  # type: ignore
        self.drag_start_camera_position = self.camera.position()
        self.drag_start_camera_target = self.camera_target
        right, up, front = self.camera.right_up_front()
        self.drag_start_camera_right = right
        self.drag_start_camera_up = up
        d = -dot(front, self.camera_world_up)
        self.drag_start_camera_pitch = acos(clamp(d, -1, 1))

    def on_rotate_press(self, position: ivec2) -> None:
        self.rotate_pressed = True
        self.pan_pressed = False
        self.start_drag(position)

    def on_rotate_release(self) -> None:
        self.rotate_pressed = False

    def on_pan_press(self, position: ivec2) -> None:
        self.pan_pressed = True
        self.rotate_pressed = False
        self.start_drag(position)

    def on_pan_release(self) -> None:
        self.pan_pressed = False

    def on_move(self, position: ivec2) -> None:
        if self.pan_pressed:
            self.pan(self.drag_start_mouse_position, position)
        if self.rotate_pressed:
            # Go back to previous camera if temporary camera is active
            if self.is_temporary_ortho:
                assert self.prev_camera_depth is not None
                assert self.prev_camera_world_up is not None

                if self.prev_camera_fov is not None:
                    self.camera = PerspectiveCamera(
                        self.camera.camera_from_world, self.prev_camera_depth, self.prev_camera_fov
                    )
                else:
                    assert isinstance(self.camera, OrthographicCamera)
                    self.camera = OrthographicCamera(
                        self.camera.camera_from_world,
                        self.prev_camera_depth,
                        self.camera.center,
                        self.camera.half_extents,
                    )
                self.camera_world_up = self.prev_camera_world_up
                # Restart drag operation with same, to adjust to the new camera after restoring world up
                self.start_drag(self.drag_start_mouse_position)

                self.prev_camera_world_up = None
                self.is_temporary_ortho = False

            if self.camera_control_mode == CameraControlMode.ORBIT:
                self.rotate_orbit(self.drag_start_mouse_position, position)
            elif self.camera_control_mode == CameraControlMode.TRACKBALL:
                self.rotate_trackball(self.drag_start_mouse_position, position)
            elif self.camera_control_mode == CameraControlMode.FIRST_PERSON:
                self.rotate_first_person(self.drag_start_mouse_position, position)
            elif (
                # self.camera_control_mode == CameraControlMode.PAN_AND_ZOOM_ORTHO or
                self.camera_control_mode == CameraControlMode.NONE
            ):
                pass
            else:
                raise RuntimeError(f"Unhandled control mode {self.camera_control_mode}")

    def on_zoom(self, scroll: dvec2) -> None:
        if scroll.y == 0:
            return

        if not self.pan_pressed and not self.rotate_pressed:
            self.zoom(scroll, False)

    def on_zoom_with_movement(self, scroll: dvec2) -> None:
        if scroll.y == 0:
            return

        if not self.pan_pressed and not self.rotate_pressed:
            self.zoom(scroll, True)

    def rotate_orbit(self, start_pos: ivec2, pos: ivec2) -> None:
        rot = self._rotation_from_mouse_delta(pos - start_pos, self.camera_target)
        position = vec3(rot * vec4(self.drag_start_camera_position, 1.0))  # type: ignore
        self.camera.camera_from_world = RigidTransform3D.look_at(
            position, self.camera_target, self.camera_world_up, self.handedness
        )

    def rotate_trackball(self, start_pos: ivec2, pos: ivec2) -> None:
        start = self.intersect_trackball(start_pos)
        current = self.intersect_trackball(pos)

        dist = distance(start, current)

        # Skip if starting and current point are too close.
        if dist < 1e-6:
            return

        # Compute axis of rotation as the vector perpendicular to the plane spanned by the
        # vectors connecting the origin to the two points.
        axis = normalize(cross(current, start))

        # Compute angle as the angle between the two vectors, if they are too far away we use the distance
        # between them instead, this makes it continue to rotate when dragging the mouse further away.
        angle = max(acos(dot(normalize(current), normalize(start))), dist)

        # Compute resulting rotation and apply it to the starting position and up vector.
        rot = translate(self.camera_target) * rotate(angle, axis) * translate(-self.camera_target)

        # Update camera
        self.camera_world_up = vec3(rot * vec4(self.drag_start_camera_up, 0.0))  # type: ignore
        new_camera_position = vec3(rot * vec4(self.drag_start_camera_position, 1.0))  # type: ignore
        self.camera.camera_from_world = RigidTransform3D.look_at(
            new_camera_position, self.camera_target, self.camera_world_up, self.handedness
        )

    def rotate_first_person(self, start_pos: ivec2, pos: ivec2) -> None:
        rot = self._rotation_from_mouse_delta(pos - start_pos, self.drag_start_camera_position)
        self.camera_target = rot * self.drag_start_camera_target  # type: ignore
        self.camera.camera_from_world = RigidTransform3D.look_at(
            self.drag_start_camera_position, self.camera_target, self.camera_world_up, self.handedness
        )

    def pan(self, start_pos: ivec2, pos: ivec2) -> None:
        delta = pos - start_pos
        if isinstance(self.camera, OrthographicCamera):
            ar_half_extents = self.camera.half_extents * vec2(self.rect.width / self.rect.height, 1)
            movement = vec2(delta) / vec2(self.rect.width, self.rect.height) * ar_half_extents * 2.0
        else:
            speed_scale = max(
                distance(self.drag_start_camera_target, self.drag_start_camera_position)
                * self.camera_pan_distance_speed_scale,
                self.camera_pan_min_speed_scale,
            )
            movement = vec2(delta) * self.camera_pan_speed * speed_scale

        delta_position = movement.x * self.drag_start_camera_right + movement.y * self.drag_start_camera_up
        self.camera_target = self.drag_start_camera_target + delta_position
        self.camera.camera_from_world = RigidTransform3D.look_at(
            self.drag_start_camera_position + delta_position, self.camera_target, self.camera_world_up, self.handedness
        )

    def zoom(self, scroll: dvec2, move: bool) -> None:
        # We move the position of the camera for perspective cameras or for temporary orthographic cameras
        # if the previous camera was a perspective camera, to ensure smooth transition back to perspective.
        if isinstance(self.camera, PerspectiveCamera) or self.prev_camera_fov is not None:
            position = self.camera.position()
            dist = distance(position, self.camera_target)
            speed_scale = max(dist * self.camera_zoom_distance_speed_scale, self.camera_zoom_min_speed_scale)
            movement = scroll.y * speed_scale * self.camera_zoom_speed
            if move:
                # If moving, also move target
                delta_position = self.camera.front() * movement
                self.camera_target += delta_position
            else:
                # If target not moving, clamp movement to keep a minimum distance to the target
                max_movement = max(dist - self.camera_zoom_min_target_distance, 0.0)
                movement = min(movement, max_movement)
                delta_position = self.camera.front() * movement

            new_position = self.camera.position() + delta_position
            self.camera.camera_from_world = RigidTransform3D.look_at(
                new_position, self.camera_target, self.camera_world_up, self.handedness
            )

        if isinstance(self.camera, OrthographicCamera):
            # TODO: maybe pan camera to ensure point under cursor stays fixed in screen coordinates?
            if self.prev_camera_fov is None:
                self.camera.half_extents -= self.camera.half_extents * self.camera_zoom_ortho_speed * scroll.y
            else:
                d = distance(new_position, self.camera_target)
                fov_radians = radians(self.prev_camera_fov)
                self.camera.half_extents = vec2(tan(fov_radians / 2) * d)

    def _rotation_from_mouse_delta(self, delta: ivec2, center_of_rotation: vec3) -> mat4:
        # Clamp pitch algorithm outline:
        # - Take current camera front
        # - Project on XZ plane (assumint Y up) -> we do this by projecting to plane defined by world_up
        # - normalize to project back to distance 1
        # - compute current angle by taking acos of projection to normal
        # - compute new angle by adding speed * y mouse movement to angle
        # - clamp angle to desired value range
        # - rotate by initial camera right with new angle
        #
        # This also works for arbitrary world up, not just axis aligned
        new_pitch = clamp(
            self.drag_start_camera_pitch - delta.y * self.camera_rotation_speed.y, math.pi * 0.01, math.pi * 0.99
        )
        delta_pitch = self.drag_start_camera_pitch - new_pitch
        rot_y = rotate(delta_pitch, self.drag_start_camera_right)
        rot_x = rotate(-self.camera_rotation_speed.x * delta.x, self.camera_world_up)
        return translate(center_of_rotation) * rot_x * rot_y * translate(-center_of_rotation)  # type: ignore

    def intersect_trackball(self, pos: ivec2) -> vec3:
        """
        Return intersection of a line passing through the mouse position at pixel coordinates x, y
        and the trackball as a point in world coordinates.
        """
        width = self.rect.width
        height = self.rect.height

        # Transform mouse coordinates from -1 to 1
        nx = 2 * (pos.x + 0.5) / width - 1
        ny = 1 - 2 * (pos.y + 0.5) / height

        # Adjust coordinates for the larger side of the viewport rectangle.
        if width > height:
            nx *= width / height
        else:
            ny *= height / width

        s = nx * nx + ny * ny
        if s <= 0.5:
            # Sphere intersection
            nz = sqrt(1 - s)
        else:
            # Hyperboloid intersection.
            nz = 1 / (2 * sqrt(s))

        # Return intersection position in world coordinates.
        return self.drag_start_camera_inverse_view_rotation * vec3(nx, ny, nz)  # type: ignore

    def set_temporary_ortho_view(self, ortho_view_type: OrthographicViewType) -> None:
        if self.pan_pressed or self.rotate_pressed:
            return

        position = self.camera.position()
        target = self.camera_target
        d = distance(position, target)
        if ortho_view_type == OrthographicViewType.POSITIVE_X or ortho_view_type == OrthographicViewType.NEGATIVE_X:
            position = target + vec3(d if ortho_view_type == OrthographicViewType.POSITIVE_X else -d, 0, 0)
            if abs(self.camera_world_up.y) > 0.5:
                up = vec3(0, 1, 0) if self.camera_world_up.y >= 0 else vec3(0, -1, 0)
            else:
                up = vec3(0, 0, 1) if self.camera_world_up.z >= 0 else vec3(0, 0, -1)
            self.temporary_ortho_grid_type = GridType.YZ_PLANE
        elif ortho_view_type == OrthographicViewType.POSITIVE_Y or ortho_view_type == OrthographicViewType.NEGATIVE_Y:
            position = target + vec3(0, d if ortho_view_type == OrthographicViewType.POSITIVE_Y else -d, 0)
            if abs(self.camera_world_up.x) > 0.5:
                up = vec3(1, 0, 0) if self.camera_world_up.x >= 0 else vec3(-1, 0, 0)
            else:
                up = vec3(0, 0, 1) if self.camera_world_up.z >= 0 else vec3(0, 0, -1)
            self.temporary_ortho_grid_type = GridType.XZ_PLANE
        elif ortho_view_type == OrthographicViewType.POSITIVE_Z or ortho_view_type == OrthographicViewType.NEGATIVE_Z:
            position = target + vec3(0, 0, d if ortho_view_type == OrthographicViewType.POSITIVE_Z else -d)
            if abs(self.camera_world_up.x) > 0.5:
                up = vec3(1, 0, 0) if self.camera_world_up.x >= 0 else vec3(-1, 0, 0)
            else:
                up = vec3(0, 1, 0) if self.camera_world_up.y >= 0 else vec3(0, -1, 0)
            self.temporary_ortho_grid_type = GridType.XY_PLANE

        if isinstance(self.camera, OrthographicCamera):
            half_extents = self.camera.half_extents
            center = self.camera.center
            if not self.is_temporary_ortho:
                self.prev_camera_fov = None
        else:
            assert isinstance(self.camera, PerspectiveCamera)
            fov_radians = radians(self.camera.fov)
            half_extents = vec2(tan(fov_radians / 2) * d)
            center = vec2(0, 0)
            if not self.is_temporary_ortho:
                self.prev_camera_fov = self.camera.fov

        if not self.is_temporary_ortho:
            self.prev_camera_depth = self.camera.depth
            self.prev_camera_world_up = self.camera_world_up
        self.camera_world_up = up

        self.camera = OrthographicCamera(
            RigidTransform3D.look_at(position, target, up, self.handedness),
            CameraDepth(0, self.camera.depth.z_far),
            center,
            half_extents,
        )

        self.is_temporary_ortho = True
