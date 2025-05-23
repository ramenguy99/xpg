from threading import Event, Thread
from queue import Queue
from typing import Callable, TypeVar
import atexit

O = TypeVar("O")

class PromiseException(RuntimeError):
    pass

class Promise:
    def __init__(self):
        self.obj: O = None
        self.exception: Exception = None
        self.event: Event = Event()
    
    def is_set(self) -> bool:
        return self.event.is_set()

    def clear(self):
        self.event.clear()
    
    def set(self, obj: O):
        self.obj = obj
        self.event.set()
    
    def set_exception(self, e: Exception):
        self.exception = e
        self.event.set()

    def get(self) -> O:
        self.event.wait()
        if self.exception:
            raise PromiseException() from self.exception
        else:
            return self.obj

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
            
            promise, func, args, kwargs = elem
            promise: Promise
            try:
                obj = func(*args, **kwargs, thread_index=thread_index)
                promise.set(obj)
            except Exception as e:
                promise.set_exception(e)
    
    def submit(self, promise: Promise, func: Callable, *args, **kwargs):
        promise.event.clear()
        self.queue.put((promise, func, args, kwargs))

    def stop(self):
        for _ in range(len(self.workers)):
            self.queue.put(None)
        for w in self.workers:
            w.join()
        self.workers = []
