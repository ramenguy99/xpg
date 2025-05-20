from threading import Event, Thread
from queue import Queue
import atexit

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
            
            # Handle exceptions in callback somehow
            elem[0](thread_index, *elem[1], **elem[2])
    
    def submit(self, func, *args, **kwargs):
        self.queue.put((func, args, kwargs))

    def stop(self):
        for _ in range(len(self.workers)):
            self.queue.put(None)
        for w in self.workers:
            w.join()
        self.workers = []
