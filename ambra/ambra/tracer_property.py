import datetime
import logging
import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from numpy.typing import DTypeLike
from pyxpg import Format

from . import tracer
from .property import (
    ImageSize,
    ListBufferProperty,
    ListImageProperty,
    ListTimeSampledAnimation,
    Property,
    PropertyItem,
    UploadSettings,
)
from .server import Client, RawMessage
from .utils.profile import profile
from .viewer import Viewer

logger = logging.getLogger(__name__)


class TracerProperty(Property, ABC):
    def __init__(self) -> None:
        self.topic: Optional[str] = None
        self.source: Optional[TracerSource] = None

    def get_frame_by_index(self, frame_index: int, thread_index: int = -1) -> PropertyItem:
        assert self.source is not None
        return self.source.get_property_frame_by_index(self, frame_index, thread_index)

    @abstractmethod
    def process(self, **kwargs: Dict[str, Any]) -> PropertyItem: ...


class TracerBufferProperty(TracerProperty, ListBufferProperty):
    def __init__(
        self, dtype: Optional[DTypeLike] = None, shape: Tuple[int, ...] = (), max_size: int = 0, name: str = ""
    ):
        ListBufferProperty.__init__(self, [], dtype, shape, max_size, None, UploadSettings(preupload=False), name)
        TracerProperty.__init__(self)


class TracerImageProperty(TracerProperty, ListImageProperty):
    def __init__(
        self, format: Optional[Format] = None, image_size: Optional[ImageSize] = None, name: str = ""
    ) -> None:
        ListImageProperty.__init__(self, [], format, image_size, None, UploadSettings(preupload=False), name)
        TracerProperty.__init__(self)


class TracerMultiProperty(TracerBufferProperty):
    def update_frame(self, frame_index: int, frame: Any) -> None:
        super().update_frame(frame_index, frame)
        self.update_subproperties(frame)

    def update(self, time: float, frame: int) -> None:
        old_frame = self.current_frame_index
        super().update(time, frame)
        if old_frame != self.current_frame_index:
            self.update_subproperties(self.get_current())

    def update_subproperties(self, frame: Any) -> None:
        pass


@dataclass
class RegisteredTopic:
    properties: List[TracerProperty] = field(default_factory=list)
    raw_timestamps: List[int] = field(default_factory=list)


class TracerSource:
    def __init__(self, timestamp_column_name: str):
        self.timestamp_column_name = timestamp_column_name

        self.registered_topics: Dict[str, RegisteredTopic] = {}
        self.registered_callbacks: Dict[str, List[Callable[[str, Dict[str, Any]], None]]] = {}
        self.first_frame_timestamp: Optional[int] = None

    def register(self, topic: str, property: TracerProperty) -> TracerProperty:
        property.source = self
        property.topic = topic
        self.registered_topics.setdefault(topic, RegisteredTopic()).properties.append(property)

        return property

    def register_callback(self, topic: str, callable: Callable[[str, Dict[str, Any]], None]) -> None:
        self.registered_callbacks.setdefault(topic, []).append(callable)

    def on_raw_message(self, viewer: Viewer, client: Client, raw_message: RawMessage) -> None:
        pass

    def on_event(self, viewer: Viewer, topic: str, msg: Dict[str, Any]) -> None:
        pass

    def start(self, first_frame_timestamp: Optional[int] = None) -> None:
        self.first_frame_timestamp = first_frame_timestamp

    def stop(self) -> None:
        pass

    @contextmanager
    def attach(self, viewer: Viewer, first_frame_timestamp: Optional[int]):  # type: ignore
        multi_properties = set()
        for reg in self.registered_topics.values():
            for p in reg.properties:
                if isinstance(p, TracerMultiProperty):
                    multi_properties.add(p)

        viewer.scene.additional_properties.extend(list(multi_properties))
        viewer.raw_message_callbacks.append(self.on_raw_message)
        self.start(first_frame_timestamp)
        try:
            yield
        finally:
            self.stop()
            viewer.raw_message_callbacks.remove(self.on_raw_message)
            viewer.scene.additional_properties = [
                p for p in viewer.scene.additional_properties if p not in multi_properties
            ]

    def get_property_frame_by_index(
        self, property: TracerProperty, frame_index: int, thread_index: int = -1
    ) -> PropertyItem:
        return super(TracerProperty, property).get_frame_by_index(frame_index, thread_index)


@dataclass
class SqliteConfig:
    journal_mode: Optional[str] = "none"
    synchronous: Optional[str] = "off"
    wal_autocheckpoint: Optional[int] = 16384
    page_size: Optional[int] = None
    cache_size: Optional[int] = None


class TracerLiveSource(TracerSource):
    def __init__(
        self,
        timestamp_column_name: str,
        collect_history: bool = True,
        collect_to_sqlite_db: bool = True,
        sqlite_db_path: Optional[Path] = None,
        persist_sqlite_db: bool = False,
        sqlite_config: Optional[SqliteConfig] = None,
    ):
        super().__init__(timestamp_column_name)

        self.collect_histroy = collect_history
        self.collect_to_sqlite_db = collect_to_sqlite_db
        self.persist_sqlite_db = persist_sqlite_db

        self.db: Optional[sqlite3.Connection] = None
        self.db_known_tables: Set[str] = set()

        if collect_to_sqlite_db:
            if sqlite_db_path is None:
                date_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
                sqlite_db_path = Path(f"tracer-live-source_{date_time}.db")
            self.db_path = sqlite_db_path

            sqlite_config = SqliteConfig() if sqlite_config is None else sqlite_config

            self.db = sqlite3.connect(self.db_path)
            if sqlite_config.journal_mode is not None:
                self.db.execute(f"PRAGMA journal_mode={sqlite_config.journal_mode}")
            if sqlite_config.synchronous is not None:
                self.db.execute(f"PRAGMA synchronous={sqlite_config.synchronous}")
            if sqlite_config.wal_autocheckpoint is not None:
                self.db.execute(f"PRAGMA wal_autocheckpoint={sqlite_config.wal_autocheckpoint}")
            if sqlite_config.page_size is not None:
                self.db.execute(f"PRAGMA page_size={sqlite_config.page_size}")
            if sqlite_config.cache_size is not None:
                self.db.execute(f"PRAGMA cache_size={sqlite_config.cache_size}")
        else:
            self.db = None

    def on_raw_message(self, viewer: Viewer, client: Client, raw_message: RawMessage) -> None:
        if raw_message.id == tracer.MessageType.TRACE_EVENT:
            topic, data = tracer.decode_trace_event(raw_message.data)
            msg = tracer.from_binary(data)
            self.on_event(viewer, topic, msg)

    def on_event(self, viewer: Viewer, topic: str, msg: Dict[str, Any]) -> None:
        callbacks = self.registered_callbacks.get(topic)
        if callbacks is not None:
            for c in callbacks:
                c(topic, msg)

        registered = self.registered_topics.get(topic)
        if registered is None:
            # Unhandled topic
            return

        # Extract and update time-related fields
        timestamp = msg[self.timestamp_column_name]
        if self.first_frame_timestamp is None:
            self.first_frame_timestamp = timestamp
        registered.raw_timestamps.append(timestamp)
        t = (timestamp - self.first_frame_timestamp) * 1e-9

        # Update max viewer playback
        if self.collect_histroy:
            viewer.playback.set_max_time(max(viewer.playback.max_time, t))

        # Serialize to sqlite if requested
        if self.collect_to_sqlite_db:
            assert self.db is not None
            if topic not in self.db_known_tables:
                tracer.create_table(self.db, topic, msg)
                clock = self.timestamp_column_name
                self.db.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS "{topic}_{clock}" ON "{topic}" ({clock})')
                self.db_known_tables.add(topic)
            with profile("Write"):
                tracer.to_sqlite(self.db, topic, **msg)
                self.db.commit()

        # Update properties
        for p in registered.properties:
            data = p.process(**msg)
            if self.collect_histroy:
                assert isinstance(p.animation, ListTimeSampledAnimation)
                p.animation.timestamps.append(t)
                if self.collect_to_sqlite_db:
                    p.num_frames += 1
                else:
                    p.append_frame(data)
            else:
                if p.num_frames == 0:
                    p.append_frame(data)
                else:
                    p.update_frame(0, data)

    def start(self, first_frame_timestamp: Optional[int] = None) -> None:
        super().start(first_frame_timestamp)

        # Ensure all properties have an empty list animation
        if self.collect_histroy:
            for registered in self.registered_topics.values():
                for p in registered.properties:
                    p.animation = ListTimeSampledAnimation([])

    def stop(self) -> None:
        if self.db is not None:
            if not self.persist_sqlite_db:
                self.db_path.unlink(missing_ok=True)
            self.db.close()
            self.db = None

    def __del__(self) -> None:
        self.stop()

    def get_property_frame_by_index(
        self, property: TracerProperty, frame_index: int, thread_index: int = -1
    ) -> PropertyItem:
        if not self.collect_histroy or not self.collect_to_sqlite_db:
            return super().get_property_frame_by_index(property, frame_index, thread_index)
        else:
            assert property.topic is not None
            assert self.db is not None

            raw_timestamps = self.registered_topics[property.topic].raw_timestamps
            with profile("Query"):
                v = tracer.from_sqlite(
                    self.db, property.topic, where=f"{self.timestamp_column_name} == {raw_timestamps[frame_index]}"
                )
            return property.process(**v)


class TracerOfflineSource(TracerSource):
    def __init__(self, timestamp_column_name: str, sqlite_db_path: Path):
        super().__init__(timestamp_column_name)

        self.db = sqlite3.connect(sqlite_db_path)

    def read_timestamps(self, db: sqlite3.Connection, topic: str) -> List[int]:
        try:
            clock = self.timestamp_column_name
            db.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS "{topic}_{clock}" ON "{topic}" ({clock})')
            data = db.execute(f'SELECT {clock} from "{topic}" ORDER BY {clock} ASC')  # noqa: S608
        except Exception as e:
            logger.info("Failed to read timestamp for topic %s: %s", topic, e)
            return []
        return [e[0] for e in data.fetchall()]

    def start(self, first_frame_timestamp: Optional[int] = None) -> None:
        assert first_frame_timestamp is not None

        super().start(first_frame_timestamp)

        for topic, registered in self.registered_topics.items():
            registered.raw_timestamps = self.read_timestamps(self.db, topic)
            for p in registered.properties:
                p.animation = ListTimeSampledAnimation(
                    [(t - first_frame_timestamp) * 1e-9 for t in registered.raw_timestamps]
                )
                p.num_frames = len(registered.raw_timestamps)

    def get_property_frame_by_index(
        self, property: TracerProperty, frame_index: int, thread_index: int = -1
    ) -> PropertyItem:
        assert property.topic is not None

        raw_timestamps = self.registered_topics[property.topic].raw_timestamps
        v = tracer.from_sqlite(
            self.db, property.topic, where=f"{self.timestamp_column_name} == {raw_timestamps[frame_index]}"
        )
        for callback in self.registered_callbacks.get(property.topic, []):
            callback(property.topic, v)
        return property.process(**v)
