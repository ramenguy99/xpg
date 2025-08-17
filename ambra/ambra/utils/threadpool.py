import atexit
from queue import Queue
from threading import Event, Thread
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar

O = TypeVar("O")


class PromiseError(RuntimeError):
    pass


class Promise(Generic[O]):
    def __init__(self) -> None:
        self.obj: Optional[O] = None
        self.exception: Optional[Exception] = None
        self.event: Event = Event()

    def is_set(self) -> bool:
        return self.event.is_set()

    def clear(self) -> None:
        self.event.clear()

    def set(self, obj: O) -> None:
        self.obj = obj
        self.event.set()

    def set_exception(self, e: Exception) -> None:
        self.exception = e
        self.event.set()

    def get(self) -> Optional[O]:
        self.event.wait()
        if self.exception is not None:
            raise PromiseError from self.exception
        return self.obj


class ThreadPool(Generic[O]):
    def __init__(self, num_workers: int):
        self.queue: Queue[
            Optional[
                Tuple[
                    Promise[O],
                    Callable[[Tuple[Any, ...], Dict[str, Any]], O],
                    Tuple[Any, ...],
                    Dict[str, Any],
                ]
            ]
        ] = Queue()
        self.workers = [
            Thread(None, self.__entry, f"ThreadPool.worker{i}", (i,), daemon=True) for i in range(num_workers)
        ]
        for w in self.workers:
            w.start()
        atexit.register(self.stop)

    def __entry(self, thread_index: int) -> None:
        while True:
            elem = self.queue.get()
            if elem is None:
                break

            promise, func, args, kwargs = elem
            try:
                # HACK: having this special thread_index variable here is a bit messy and mypy complains about it, maybe we can do something cleaner
                obj = func(*args, **kwargs, thread_index=thread_index)  # type: ignore
                promise.set(obj)
            except Exception as e:
                promise.set_exception(e)

    def submit(
        self,
        promise: Promise[O],
        func: Callable[[Tuple[Any, ...], Dict[str, Any]], O],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        promise.event.clear()
        self.queue.put((promise, func, args, kwargs))

    def stop(self) -> None:
        for _ in range(len(self.workers)):
            self.queue.put(None)
        for w in self.workers:
            w.join()
        self.workers = []
