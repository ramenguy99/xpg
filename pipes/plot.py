import matplotlib.pyplot as plt
import matplotlib.patches as patches

from colors import ColorInfo
from samples import *
from render import RenderParams, render, ArrowStyle

fig = plt.figure(figsize=(20, 9))
ax = fig.gca()

params = RenderParams()

def rect(text, x, y, w, h, color: ColorInfo):
    y = params.height - y - h
    r = patches.Rectangle((x, y), w, h, facecolor=color.face_color, edgecolor=color.border_color)
    ax.add_patch(r)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center')

def text(text, x, y):
    y = params.height - y
    ax.text(x, y, text, ha='center', va='center')

def arrow(x0, y0, x1, y1, style: ArrowStyle):
    if style == ArrowStyle.DASHED:
        linestyle = "--"
    elif style == ArrowStyle.SOLID:
        linestyle = "-"
    else:
        raise ValueError(f"Unhandled style {style}")

    y0 = params.height - y0
    y1 = params.height - y1
    a = patches.FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='->', linestyle=linestyle, mutation_scale=15, color='black')
    ax.add_patch(a)

p = double_buffered_async_gpu_upload_sync_submit_half_rate()

rendered = render(p, params)
for t in rendered.texts:
    text(t.text, t.x, t.y)
for a in rendered.arrows:
    arrow(a.x0, a.y0, a.x1, a.y1, a.style)
for r in rendered.rects:
    rect(r.text, r.x, r.y, r.w, r.h, r.color)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set_xlim(0, params.width)
ax.set_ylim(0, params.height)
ax.set_aspect('equal')
plt.show()