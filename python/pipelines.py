import watchfiles
from queue import Queue
from typing import List, Callable, Dict
from dataclasses import dataclass
from threading import Thread
from pathlib import Path

@dataclass
class Pipeline:
    callback: Callable
    deps: List[str]
    refresh: bool = True

class PipelineCache:
    def __init__(self, pipelines: List[Pipeline]):
        self.queue = Queue()
        self.reload: Dict[Path, List[Pipeline]] = {}
        for pipe in pipelines:
            for dep in pipe.deps:
                self.reload.setdefault(Path(dep).absolute(), []).append(pipe)
            self.queue.put(pipe)
        self.should_stop = False
        self.thread = Thread(target=self.__thread_entry)
        self.thread.start()

    def __thread_entry(self):
        if not self.reload.keys():
            return

        for changes in watchfiles.watch(*self.reload.keys(), debounce=200):
            if self.should_stop:
                break

            for change, path in changes:
                if change == watchfiles.Change.modified:
                    for pipe in self.reload[Path(path).absolute()]:
                        if not pipe.refresh:
                            pipe.refresh = True
                            self.queue.put(pipe)

    def refresh(self):
        while not self.queue.empty():
            pipe = self.queue.get()
            pipe.callback()
            pipe.refresh = False
    
    def stop(self):
        if self.reload.keys():
            self.should_stop = True
            for p in self.reload.keys():
                p.touch()
                break
        self.thread.join()
