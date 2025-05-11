from dataclasses import dataclass
from colors import ColorInfo
from typing import List
from enum import Enum, auto
from pipelines import Pipeline, Resource

@dataclass
class Text:
    text: str
    x: int
    y: int

@dataclass
class Rect:
    text: str
    x: int
    y: int
    w: int
    h: int
    color: ColorInfo

class ArrowStyle(Enum):
    SOLID = 0
    DASHED = 1

@dataclass
class Arrow:
    x0: int
    y0: int
    x1: int
    y1: int
    style: ArrowStyle

@dataclass
class Rendered:
    texts: List[Rect]
    rects: List[Rect]
    arrows: List[Arrow]

@dataclass
class RenderParams:
    width = 2000
    height = 900

    thread_height = 100
    thread_offset_x = 50
    thread_offset_y = 50

    task_height = 50
    task_offset_x = 120
    task_offset_y = 25

    width_per_second = 50

def render(p: Pipeline, params: RenderParams) -> Rendered:
    texts: List[Text] = []
    rects: List[Rect] = []
    arrows: List[Arrow] = []

    for i, t in enumerate(p.threads.values()):
        texts.append(Text(t.name, params.thread_offset_x, params.thread_height * i + params.thread_offset_y))

    max_iters = 10
    for i in range(max_iters):
        progress = False

        for thread_idx, thread in enumerate(p.threads.values()):
            for task in thread.tasks:
                ready = True
                for q in task.resources.values():
                    if q.empty():
                        ready = False
                if not ready:
                    continue

                progress = True

                start = thread.pos

                # print(thread.name, task.name, start, "START")
                res = []
                for k, q in task.resources.items():
                    r: Resource = q.get()

                    # print(thread.name, task.name, r.start, f"R({k}): {len(res)}")
                    start = max(r.start, start)
                    res.append(r)

                rects.append(Rect(task.name, params.task_offset_x + start * params.width_per_second, params.task_offset_y + thread_idx * params.thread_height, task.duration * params.width_per_second, params.task_height, task.color))

                for r in res:
                    if r.producer_thread_idx is not None:
                        if r.producer_thread_idx < thread_idx:
                            arrows.append(Arrow(params.task_offset_x +  r.start * params.width_per_second, params.task_offset_y + r.producer_thread_idx * params.thread_height + params.task_height, params.task_offset_x + start * params.width_per_second, thread_idx * params.thread_height + params.task_offset_y, ArrowStyle.SOLID))
                        elif r.producer_thread_idx > thread_idx:
                            arrows.append(Arrow(params.task_offset_x +  r.start * params.width_per_second, params.task_offset_y + r.producer_thread_idx * params.thread_height, params.task_offset_x + start * params.width_per_second, thread_idx * params.thread_height + params.task_offset_y + params.task_height, ArrowStyle.DASHED))

                thread.pos = start + task.duration
                for out in task.outputs:
                    for i in range(0, (out.ratio[0] + out.counter % out.ratio[1]) // out.ratio[1]):
                        out.task.resources[out.resource_name].put(Resource(start + task.duration, thread_idx))
                    out.counter += 1

        if not progress:
            break

    return Rendered(texts, rects, arrows)
