from typing import Callable, Optional
from enum import Enum, auto
from dataclasses import dataclass
from threading import Event, Thread
from queue import Queue
import atexit

from multiprocessing.pool import ThreadPool

class State(Enum):
    EMPTY = auto()
    FILLING = auto()
    FULL = auto()


@dataclass
class Frame:
    obj: Optional[object]
    state: State
    event: Event

def do_work(thread_index: int, frame: Frame, load: Callable[[int], object], buffer_index: int, frame_index: int):
    frame.obj = load(thread_index, buffer_index, frame_index)
    frame.state = State.FULL
    frame.event.set()

class ThreadPool:
    def __init__(self, num_workers: int):
        self.queue = Queue()
        self.workers = [Thread(None, self.__entry, f"ThreadPool.worker{i}", (i,), daemon=True) for i in range(num_workers) ]
        for w in self.workers:
            w.start()
        atexit.register(self.stop)
    
    def __entry(self, thread_index):
        while True:
            elem = self.queue.get()
            if elem is None:
                break
            elem[0](thread_index, *elem[1], **elem[2])
    
    def submit(self, func, *args, **kwargs):
        self.queue.put((func, args, kwargs))

    def stop(self):
        for _ in range(len(self.workers)):
            self.queue.put(None)
        for w in self.workers:
            w.join()
        self.workers = []

class BufferedStream:
    def __init__(self, n: int, buffer_length: int, num_workers: int, load: Callable[[int], object]):
        self.load = load
        self.n = n
        self.buffer = [Frame(None, State.EMPTY, Event()) for _ in range(buffer_length)]
        self.cursor = 0
        self.buffer_cursor = 0

        self.pool = ThreadPool(num_workers)

        for i in range(buffer_length):
            self.enqueue_load(i, i)
    
    def enqueue_load(self, buffer_index: int, frame_index: int):
        element = self.buffer[buffer_index]
        if element.state == State.FILLING:
            element.event.wait()
        element.state = State.FILLING
        element.event.clear()

        self.pool.submit(do_work, element, self.load, buffer_index, frame_index)
    
    def get_frame(self, frame: int) -> object:
        if frame >= self.n:
            return None
        
        # Frame in range [cursor, cursor + stream length]
        normalized_frame = frame + self.n if frame < self.cursor else frame
        reset = False
        if normalized_frame >= self.cursor + len(self.buffer):
            normalized_frame = frame
            
            self.cursor = frame
            self.buffer_cursor = 0
            reset = True

            self.enqueue_load(0, frame)

        # Wait for frame to be done 
        delta = normalized_frame - self.cursor
        buffer_index = (self.buffer_cursor + delta) % len(self.buffer)

        element = self.buffer[buffer_index]
        if element.state == State.FILLING:
            element.event.wait()
        element.event.wait()

        obj = element.obj

        if reset:
            for i in range(1, len(self.buffer)):
                self.enqueue_load(i, frame + 1)
        else:
            for i in range(delta):
                self.enqueue_load(
                    (self.buffer_cursor + i) % len(self.buffer),
                    (frame + len(self.buffer) + i) % self.n,
                )
            self.cursor = frame
            self.buffer_cursor = buffer_index
        return obj

    def stop(self):
        self.pool.stop()