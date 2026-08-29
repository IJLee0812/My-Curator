"""Two cameras on the ego vehicle, writing raw frames for the encoder.

The world runs in synchronous mode at the render tick, so one ``world.tick()`` yields
exactly one image per camera and the recording cannot drift from simulation time. Frames
land in a fixed-size raw stream per view, which makes a segment a byte range rather than a
seek — the encoder relies on that.

The warm-up is simulated but not recorded, so a synthetic view and its source segment have
the same number of frames.

CARLA is imported lazily so this module stays importable on a host with no simulator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any

log = logging.getLogger(__name__)

CAMERA_BLUEPRINT = "sensor.camera.rgb"


class RecordingError(RuntimeError):
    """The cameras did not deliver the frames the segment needs."""


@dataclass
class RecordedView:
    view: str
    path: Path
    width: int
    height: int
    frames: int


class DualViewRecorder:
    """Attach the rig, drive the clock, write one raw stream per view."""

    def __init__(self, world: Any, ego: Any, cameras: list[dict[str, Any]], out_dir: Path) -> None:
        self._world = world
        self._ego = ego
        self._cameras = cameras
        self._out_dir = out_dir
        self._sensors: list[Any] = []
        self._queues: dict[str, Queue] = {}
        self._handles: dict[str, Any] = {}
        self._counts: dict[str, int] = {}

    def __enter__(self) -> DualViewRecorder:
        self._attach()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _attach(self) -> None:
        import carla

        library = self._world.get_blueprint_library()
        for camera in self._cameras:
            blueprint = library.find(CAMERA_BLUEPRINT)
            blueprint.set_attribute("image_size_x", str(camera["image_size_x"]))
            blueprint.set_attribute("image_size_y", str(camera["image_size_y"]))
            blueprint.set_attribute("fov", str(camera["fov"]))
            # sensor_tick is deliberately left alone: the world already runs synchronously
            # at the render tick, and setting it to the same value makes the capture
            # interval a borderline comparison — the camera then delivers one frame only.
            for name, value in (camera.get("attributes") or {}).items():
                if blueprint.has_attribute(name):
                    blueprint.set_attribute(name, str(value))

            x, y, z, roll, pitch, yaw = camera["transform"]
            transform = carla.Transform(
                carla.Location(x=x, y=y, z=z), carla.Rotation(roll=roll, pitch=pitch, yaw=yaw)
            )
            sensor = self._world.spawn_actor(blueprint, transform, attach_to=self._ego)

            view = camera["view"]
            queue: Queue = Queue()
            sensor.listen(queue.put)
            self._sensors.append(sensor)
            self._queues[view] = queue
            self._counts[view] = 0
            self._handles[view] = (self._out_dir / f"{view}.bgra").open("wb")

    def tick(self, *, record: bool) -> None:
        """Advance one frame, writing what the cameras produced only when recording."""
        self._world.tick()
        for view, queue in self._queues.items():
            try:
                image = queue.get(timeout=5.0)
            except Empty as exc:
                raise RecordingError(f"camera {view!r} delivered no frame for this tick") from exc
            if record:
                self._handles[view].write(bytes(image.raw_data))
                self._counts[view] += 1

    def close(self) -> None:
        for sensor in self._sensors:
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:
                log.debug("sensor already gone")
        self._sensors.clear()
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def views(self, expected_frames: int) -> list[RecordedView]:
        recorded = []
        for camera in self._cameras:
            view = camera["view"]
            count = self._counts[view]
            if count < expected_frames:
                raise RecordingError(
                    f"camera {view!r} recorded {count} of {expected_frames} frames"
                )
            recorded.append(
                RecordedView(
                    view=view,
                    path=self._out_dir / f"{view}.bgra",
                    width=camera["image_size_x"],
                    height=camera["image_size_y"],
                    frames=count,
                )
            )
        return recorded
