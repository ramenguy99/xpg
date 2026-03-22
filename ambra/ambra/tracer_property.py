
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import datetime
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional

from ambra import tracer
from ambra.property import ListTimeSampledAnimation, Property, ListBufferProperty, ListImageProperty, UploadSettings, PropertyItem
from ambra.viewer import Viewer
from ambra.server import RawMessage, Client
from ambra.utils.profile import profile

class TracerProperty(Property):
    def __init__(self):
        self.topic: Optional[str] = None
        self.source: TracerSource = None

    def process(self, **kwargs) -> PropertyItem:
        pass

    def get_frame_by_index(self, frame_index: int, thread_index: int = -1):
        return self.source.get_property_frame_by_index(self, frame_index, thread_index)


class TracerBufferProperty(TracerProperty, ListBufferProperty):
    def __init__(self, dtype = None, shape = (), max_size = 0, name = ""):
        ListBufferProperty.__init__(self, [], dtype, shape, max_size, None, UploadSettings(preupload=False), name)
        TracerProperty.__init__(self)


class TracerImageProperty(TracerProperty, ListImageProperty):
    def __init__(self, format = None, image_size = None, name = ""):
        ListImageProperty.__init__(self, [], format, image_size, None, UploadSettings(preupload=False), name)
        TracerProperty.__init__(self)


class TracerMultiProperty(TracerBufferProperty):
    def update_frame(self, frame_index, frame):
        super().update_frame(frame_index, frame)
        self.update_subproperties(frame)

    def update(self, time, frame):
        old_frame = self.current_frame_index
        super().update(time, frame)
        if old_frame != self.current_frame_index:
            self.update_subproperties(self.get_current())

    def update_subproperties(self, frame):
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

    def register_callback(self, topic: str, callable: Callable[[str, Dict[str, Any]], None]):
        self.registered_callbacks.setdefault(topic, []).append(callable)

    def on_raw_message(self, viewer: Viewer, client: Client, raw_message: RawMessage):
        pass

    def on_event(self, viewer: Viewer, topic: str, msg: Dict[str, Any]):
        pass

    def start(self, first_frame_timestamp: Optional[int] = None):
        self.first_frame_timestamp = first_frame_timestamp

    def stop(self) -> None:
        pass

    @contextmanager
    def attach(self, viewer: Viewer, first_frame_timestamp: Optional[int]):
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
            viewer.scene.additional_properties = [p for p in viewer.scene.additional_properties if p not in multi_properties]

    def get_property_frame_by_index(self, property: TracerProperty, frame_index: int, thread_index: int = -1):
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
        sqlite_config: Optional[SqliteConfig] = None
    ):
        super().__init__(timestamp_column_name)

        self.collect_histroy = collect_history
        self.collect_to_sqlite_db = collect_to_sqlite_db
        self.persist_sqlite_db = persist_sqlite_db

        if collect_to_sqlite_db:
            if sqlite_db_path is None:
                date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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

            self.db_known_tables = set()

    def on_raw_message(self, viewer: Viewer, client: Client, raw_message: RawMessage):
        if raw_message.id == tracer.MessageType.TRACE_EVENT:
            topic, data = tracer.decode_trace_event(raw_message.data)
            msg = tracer.from_binary(data)
            self.on_event(viewer, topic, msg)

    def on_event(self, viewer: Viewer, topic: str, msg: Dict[str, Any]):
        callbacks = self.registered_callbacks.get(topic)
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

    def start(self, first_frame_timestamp: Optional[int]) -> None:
        super().start(first_frame_timestamp)

        # Ensure all properties have an empty list animation
        if self.collect_histroy:
            for topic, registered in self.registered_topics.items():
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

    def get_property_frame_by_index(self, property: TracerProperty, frame_index: int, thread_index: int = -1):
        if not self.collect_histroy or not self.collect_to_sqlite_db:
            return super().get_property_frame_by_index(property, frame_index, thread_index)
        else:
            raw_timestamps = self.registered_topics[property.topic].raw_timestamps
            with profile("Query"):
                v = tracer.from_sqlite(self.db, property.topic, where=f'{self.timestamp_column_name} == {raw_timestamps[frame_index]}')
            b = property.process(**v)
            return b

class TracerOfflineSource(TracerSource):
    def __init__(self, timestamp_column_name: str, sqlite_db_path: Path):
        super().__init__(timestamp_column_name)

        self.db = sqlite3.connect(sqlite_db_path)

    def read_timestamps(self, db: sqlite3.Cursor, topic: str) -> None:
        try:
            clock = self.timestamp_column_name
            db.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS "{topic}_{clock}" ON "{topic}" ({clock})')
            data = db.execute(f'SELECT {clock} from "{topic}" ORDER BY {clock} ASC')
        except Exception as e:
            print(f"Failed to read timestamp for topic {topic}: {e}")
            return []
        return [e[0] for e in data.fetchall()]

    def start(self, first_frame_timestamp: int) -> None:
        super().start(first_frame_timestamp)

        for topic, registered in self.registered_topics.items():
            registered.raw_timestamps = self.read_timestamps(self.db, topic)
            for p in registered.properties:
                p.animation = ListTimeSampledAnimation([(t - first_frame_timestamp) * 1e-9 for t in registered.raw_timestamps])
                p.num_frames = len(registered.raw_timestamps)

    def get_property_frame_by_index(self, property: TracerProperty, frame_index: int, thread_index: int = -1):
        raw_timestamps = self.registered_topics[property.topic].raw_timestamps
        v = tracer.from_sqlite(self.db, property.topic, where=f'{self.timestamp_column_name} == {raw_timestamps[frame_index]}')
        for callback in self.registered_callbacks.get(property.topic, []):
            callback(property.topic, v)
        b = property.process(**v)
        return b
