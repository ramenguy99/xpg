import signal
import time

import numpy as np

from ambra import tracer

WIDTH = 64
HEIGHT = 48


tracer.init_tracer()
tcp_sub = tracer.TcpSubscriber(
    host="127.0.0.1",
    port=int(9168),
    patterns=[r".*"],
    connect_timeout=5.0,
    reconnect_delay=1.0,
    queue_size=4,
    queue_full_policy=tracer.QueueFullPolicy.DROP,
)
tracer.register_subscriber(tcp_sub)

sqlite_sub = tracer.SqliteSubscriber(
    db_path="client.db",
    patterns=[r".*"],
    queue_size=256,
    queue_full_policy=tracer.QueueFullPolicy.DROP,
)
tracer.register_subscriber(sqlite_sub)

start_t = time.monotonic_ns()

x, y = np.meshgrid(np.arange(WIDTH, dtype=np.float32), np.arange(HEIGHT, dtype=np.float32))
x = x.flatten() / WIDTH * 2.0
y = y.flatten() / HEIGHT * 2.0

stop = False


def signal_handler(sig, frame):
    global stop
    stop = True


signal.signal(signal.SIGINT, signal_handler)

max_q_size = 0
while not stop:
    ts = time.monotonic_ns()
    t = (ts - start_t) * 1e-9 * 2
    z = np.sin(x * 5 + t) * np.cos(y * 5 + t) * 0.5 + 0.5
    pts = np.vstack((x, y, z), dtype=np.float32).T

    tracer.trace("/points", monotonic_ts=ts, points=pts)
    dt = time.monotonic_ns() - ts

    print(
        "TCP",
        ts,
        tcp_sub._send_queue.qsize(),
        dt * 1e-6,
        tcp_sub._serialized_bytes,
        tcp_sub._enqueued_events,
        tcp_sub._dropped_events,
    )
    print(
        "SQL",
        ts,
        sqlite_sub._queue.qsize(),
        max_q_size,
        dt * 1e-6,
        sqlite_sub._enqueued_events,
        sqlite_sub._dropped_events,
    )

    max_q_size = max(sqlite_sub._queue.qsize(), max_q_size)

    time.sleep(0.001)

tracer.close_tracer()
