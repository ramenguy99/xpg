import numpy as np

from ambra.config import Config, GuiConfig
from ambra.viewer import Viewer
from ambra.widgets import ImageInspector

viewer = Viewer(
    config=Config(
        gui=GuiConfig(
            stats=True,
        ),
    ),
)


H, W = 8, 16
img_data = np.zeros((H, W, 4))
for i in range(H):
    for j in range(W):
        img_data[i, j, 0] = (i + 0.5) / H
        img_data[i, j, 1] = (j + 0.5) / W
        img_data[i, j, 3] = 1
img_data = (img_data * 255.0).astype(np.uint8)

img_inspector = ImageInspector("Image Inspector", img_data)

viewer.scene.widgets.append(img_inspector)

viewer.run()
