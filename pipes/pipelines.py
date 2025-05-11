from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from queue import Queue

@dataclass
class Resource:
    start: int
    producer_thread_idx: Optional[int] = None

@dataclass
class Output:
    task: "Task"
    resource_name: str
    ratio: Tuple[int, int]
    counter: int

@dataclass
class Task:
    name: str
    duration: int
    color: str

    resources: Dict[str, Queue] = field(default_factory=dict)
    outputs: List[Output] = field(default_factory=list)

@dataclass
class Thread:
    name: str
    tasks: List[Task]
    pos: int = 0

@dataclass
class Pipeline:
    threads: Dict[str, Thread]

    def _task(self, thread_name: str, task_name: str) -> Task:
        thread = self.threads[thread_name]
        for t in thread.tasks:
            if t.name == task_name:
                return t
        assert False, task_name

    def add(self, thread_name: str, task_name: str, resource_name: str, count=1):
        task = self._task(thread_name, task_name)
        for _ in range(count):
            task.resources.setdefault(resource_name, Queue()).put(Resource(0))

    def produces(self, frm: Tuple[str, str], to: Tuple[str, str], resource_name: str, ratio=(1, 1)):
        frm_task = self._task(*frm)
        to_task = self._task(*to)
        frm_task.outputs.append(Output(to_task, resource_name, ratio, ratio[1] - 1)) 
        to_task.resources.setdefault(resource_name, Queue())
