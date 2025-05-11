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

W = 2000
H = 900


fig = plt.figure(figsize=(20, 9))
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

if False:
    # Double buffered async CPU load
    threads = {
        "load": Thread("CPU - LOAD", [Task("Load", 1, ORANGE)]),
        "cpu": Thread("CPU - MAIN", [Task("Commands", 3, YELLOW)]),
        "gpu": Thread("GPU - DRAW", [Task("Draw", 3, GREEN), Task("Present", 1.5, VIOLET)]),
    }
    p = Pipeline(threads)

    N = 2
    p.add("load", "Load", "cpubuf", count=N)
    p.add("cpu", "Commands", "cmd", count=N)
    p.produces(("load", "Load"), ("cpu", "Commands"), "cpubuf")
    p.produces(("cpu", "Commands"), ("load", "Load"), "cpubuf")
    p.produces(("cpu", "Commands"), ("gpu", "Draw"), "cmd")
    p.produces(("gpu", "Draw"), ("gpu", "Present"), "img")
    p.produces(("gpu", "Present"), ("cpu", "Commands"), "cmd")

if False:
    # Double buffered async GPU upload
    threads = {
        "load": Thread("CPU - LOAD", [Task("Load", 2, ORANGE)]),
        "cpu": Thread("CPU - MAIN", [Task("Commands", 2, YELLOW)]),
        "gpu": Thread("GPU - DRAW", [Task("Draw", 2.75, GREEN), Task("Present", 1.25, VIOLET)]),
        "copy": Thread("GPU - COPY", [Task("Upload", 2, BLUE)]),
    }
    p = Pipeline(threads)

    N = 2
    p.add("load", "Load", "cpubuf", count=N)
    p.add("cpu", "Commands", "cmd", count=N)
    p.add("copy", "Upload", "gpubuf", count=N)

    p.produces(("load", "Load"), ("copy", "Upload"), "cpubuf")
    p.produces(("gpu", "Draw"), ("copy", "Upload"), "gpubuf")
    p.produces(("copy", "Upload"), ("gpu", "Draw"), "gpubuf")
    p.produces(("cpu", "Commands"), ("gpu", "Draw"), "cmd")
    p.produces(("gpu", "Draw"), ("gpu", "Present"), "img")
    p.produces(("gpu", "Draw"), ("load", "Load"), "cpubuf")
    p.produces(("gpu", "Present"), ("cpu", "Commands"), "cmd")

if False:
    # 
    threads = {
        "load": Thread("CPU - LOAD", [Task("Load", 4, ORANGE)]),
        "cpu": Thread("CPU - MAIN", [Task("CopyCmd", 1, ORANGE), Task("Cmd", 1, YELLOW)]),
        "gpu": Thread("GPU - DRAW", [Task("Draw", 3, GREEN), Task("Present", 1, VIOLET)]),
        "copy": Thread("GPU - COPY", [Task("Upload", 5, BLUE)]),
    }
    p = Pipeline(threads)

    p.add("load", "Load", "cpubuf", count=3)
    p.add("cpu", "CopyCmd", "cmd", count=2)
    p.add("copy", "Upload", "gpubuf", count=2)

    p.produces(("load", "Load"), ("cpu", "CopyCmd"), "copybuf")
    p.produces(("cpu", "CopyCmd"), ("copy", "Upload"), "copybuf")
    p.produces(("cpu", "CopyCmd"), ("cpu", "Cmd"), "exec")
    p.produces(("gpu", "Present"), ("cpu", "CopyCmd"), "cmd")
    p.produces(("copy", "Upload"), ("gpu", "Draw"), "copybuf")
    p.produces(("gpu", "Draw"), ("gpu", "Present"), "img")
    p.produces(("gpu", "Draw"), ("copy", "Upload"), "gpubuf")
    p.produces(("cpu", "Cmd"), ("gpu", "Draw"), "cmd")
    p.produces(("gpu", "Present"), ("load", "Load"), "cpubuf")

if True:
    # 
    threads = {
        "load": Thread("CPU - LOAD", [Task("Load", 4, ORANGE)]),
        "cpu": Thread("CPU - MAIN", [Task("Copy", 0.1, YELLOW), Task("Cmd", 0.1, YELLOW)]),
        "gpu": Thread("GPU - DRAW", [Task("Draw", 3, GREEN), Task("Present", 1, VIOLET)]),
        "copy": Thread("GPU - COPY", [Task("Upload", 5, BLUE)]),
    }
    p = Pipeline(threads)

    N = 2
    F = 2
    p.add("load", "Load", "cpubuf", count=N)
    p.add("cpu", "Copy", "cmd", count=N)
    # p.add("copy", "Upload", "gpubuf", count=N)

    p.produces(("load", "Load"), ("cpu", "Copy"), "copybuf", (F, 1))
    p.produces(("cpu", "Copy"), ("copy", "Upload"), "copybuf", (1, F))
    p.produces(("cpu", "Copy"), ("cpu", "Cmd"), "exec")
    p.produces(("gpu", "Present"), ("cpu", "Copy"), "cmd")
    p.produces(("copy", "Upload"), ("gpu", "Draw"), "copybuf", (F, 1))
    p.produces(("gpu", "Draw"), ("gpu", "Present"), "img")
    # p.produces(("gpu", "Draw"), ("copy", "Upload"), "gpubuf", (1, F))
    p.produces(("cpu", "Cmd"), ("gpu", "Draw"), "cmd")
    p.produces(("gpu", "Present"), ("load", "Load"), "cpubuf", (1, F))

for i, t in enumerate(threads.values()):
    text(t.name, THREAD_OFFSET_X, THREAD_HEIGHT * i + THREAD_OFFSET_Y)

max_iters = 10
for i in range(max_iters):
    progress = False

    for thread_idx, thread in enumerate(threads.values()):
        for task in thread.tasks:
            ready = True
            for q in task.resources.values():
                if q.empty():
                    ready = False
            if not ready:
                continue

            progress = True

            start = thread.pos

            print(thread.name, task.name, start, "START")
            res = []
            for k, q in task.resources.items():
                r: Resource = q.get()

                print(thread.name, task.name, r.start, f"R({k}): {len(res)}")

                start = max(r.start, start)
                res.append(r)

            rect(task.name, TASK_OFFSET_X + start * WIDTH_PER_SECOND, TASK_OFFSET_Y + thread_idx * THREAD_HEIGHT, task.duration * WIDTH_PER_SECOND, TASK_HEIGHT, task.color)

            for r in res:
                if r.producer_thread_idx is not None:
                    if r.producer_thread_idx < thread_idx:
                        arrow(TASK_OFFSET_X +  r.start * WIDTH_PER_SECOND, TASK_OFFSET_Y + r.producer_thread_idx * THREAD_HEIGHT + TASK_HEIGHT, TASK_OFFSET_X + start * WIDTH_PER_SECOND, thread_idx * THREAD_HEIGHT + TASK_OFFSET_Y)
                    elif r.producer_thread_idx > thread_idx:
                        arrow(TASK_OFFSET_X +  r.start * WIDTH_PER_SECOND, TASK_OFFSET_Y + r.producer_thread_idx * THREAD_HEIGHT, TASK_OFFSET_X + start * WIDTH_PER_SECOND, thread_idx * THREAD_HEIGHT + TASK_OFFSET_Y + TASK_HEIGHT, linestyle="--")

            thread.pos = start + task.duration
            for out in task.outputs:
                for i in range(0, (out.ratio[0] + out.counter % out.ratio[1]) // out.ratio[1]):
                    out.task.resources[out.resource_name].put(Resource(start + task.duration, thread_idx))
                out.counter += 1

    if not progress:
        break

# plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect('equal')
plt.show()
