import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
from queue import Queue

@dataclass
class ColorInfo:
    face_color: str
    border_color: str

YELLOW = ColorInfo(face_color="#FFF2CC", border_color="#D6B656")
GREEN = ColorInfo(face_color="#D5E8D4", border_color="#82B366")
VIOLET = ColorInfo(face_color="#E1D5E7", border_color="#9673A6")
BLUE = ColorInfo(face_color="#DAE8FC", border_color="#6C8EBF")
ORANGE = ColorInfo(face_color="#FFE6CC", border_color="#D79B00")

W = 1600
H = 900


fig = plt.figure(figsize=(16, 9))
ax = fig.gca()

def rect(text, x, y, w, h, color: ColorInfo):
    y = H - y - h
    r = patches.Rectangle((x, y), w, h, facecolor=color.face_color, edgecolor=color.border_color)
    ax.add_patch(r)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center')

def text(text, x, y):
    y = H - y
    ax.text(x, y, text, ha='center', va='center')

def arrow(x0, y0, x1, y1, linestyle="-"):
    y0 = H - y0
    y1 = H - y1
    a = patches.FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='->', linestyle=linestyle, mutation_scale=15, color='black')
    ax.add_patch(a)

# rect("Hello", 10, 10, 40, 20, "blue")
# rect("Hi", 10, 30, 40, 20, "red")
# arrow(10, 50, 10, 70)

THREAD_HEIGHT = 100
THREAD_OFFSET_X = 50
THREAD_OFFSET_Y = 50

TASK_HEIGHT = 50
TASK_OFFSET_X = 120
TASK_OFFSET_Y = 25

WIDTH_PER_SECOND = 50

@dataclass
class Resource:
    start: int
    producer_thread_idx: Optional[int] = None

@dataclass
class Task:
    name: str
    duration: int
    color: str

    resources: Queue = field(default_factory=Queue)
    outputs: List["Task"] = field(default_factory=list)

    # resources: Dict[str, Queue] = field(default_factory=Queue)
    # outputs: List[("Task", str)] = field(default_factory=list)

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
        assert False, t.name

    def add(self, thread_name: str, task_name: str, count=1):
        task = self._task(thread_name, task_name)
        for _ in range(count):
            task.resources.put(Resource(0))

    def edge(self, frm: Tuple[str, str], to: Tuple[str, str]):
        frm_task = self._task(*frm)
        to_task = self._task(*to)
        frm_task.outputs.append(to_task) 

threads = {
    "load": Thread("CPU - LOAD", [Task("Load", 1, ORANGE)]),
    "cpu": Thread("CPU - MAIN", [Task("Commands", 3, YELLOW)]),
    # Thread("GPU - DRAW", [Task("Draw", 2, GREEN)]),
    "gpu": Thread("GPU - DRAW", [Task("Draw", 3, GREEN), Task("Present", 1.5, VIOLET)]),
    # Thread("GPU - COPY", [Task("Upload", 4, BLUE)]),
}

p = Pipeline(threads)
p.add("load", "Load", count=2)

p.edge(("load", "Load"), ("cpu", "Commands"))
p.edge(("cpu", "Commands"), ("load", "Load"))
p.edge(("cpu", "Commands"), ("gpu", "Draw"))
p.edge(("gpu", "Draw"), ("gpu", "Present"))
# p.edge(("gpu", "Present"), ("cpu", "Commands"))


for i, t in enumerate(threads.values()):
    text(t.name, THREAD_OFFSET_X, THREAD_HEIGHT * i + THREAD_OFFSET_Y)

max_iters = 5
for i in range(max_iters):
    progress = False

    for thread_idx, thread in enumerate(threads.values()):
        for task in thread.tasks:
            if task.resources.empty():
                continue
            progress = True

            r: Resource = task.resources.get()
            start = max(r.start, thread.pos)
            rect(task.name, TASK_OFFSET_X + start * WIDTH_PER_SECOND, TASK_OFFSET_Y + thread_idx * THREAD_HEIGHT, task.duration * WIDTH_PER_SECOND, TASK_HEIGHT, task.color)
            if r.producer_thread_idx is not None:
                if r.producer_thread_idx < thread_idx:
                    arrow(TASK_OFFSET_X +  r.start * WIDTH_PER_SECOND, TASK_OFFSET_Y + r.producer_thread_idx * THREAD_HEIGHT + TASK_HEIGHT, TASK_OFFSET_X + start * WIDTH_PER_SECOND, thread_idx * THREAD_HEIGHT + TASK_OFFSET_Y)
                elif r.producer_thread_idx > thread_idx:
                    arrow(TASK_OFFSET_X +  r.start * WIDTH_PER_SECOND, TASK_OFFSET_Y + r.producer_thread_idx * THREAD_HEIGHT, TASK_OFFSET_X + start * WIDTH_PER_SECOND, thread_idx * THREAD_HEIGHT + TASK_OFFSET_Y + TASK_HEIGHT, linestyle="--")


            thread.pos = start + task.duration
            for out in task.outputs:
                out.resources.put(Resource(start + task.duration, thread_idx))

    if not progress:
        break

# plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect('equal')
plt.show()
