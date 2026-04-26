from ambra.utils.descriptors import create_descriptor_layout_pool_and_set
import numpy as np
from pyxpg import AllocType, DescriptorSetBinding, DescriptorType, Format, Image, ImageLayout, ImageUsageFlags, Stage, imgui
from pyxpg.imgui import implot, Vec4
from pyxpg.imgui.implot import Marker

from ambra.config import Config, GuiConfig
from ambra.utils.hook import hook
from ambra.viewer import Viewer

data = np.array(
    [
        [1, 5, 3, 2],
        [10, 50, 30, 20],
        [100, 500, 300, 200],
    ],
    np.int16,
)
colors = np.array(
    [
        0xff0000ff,
        0xff00ff00,
        0xffff0000,
    ],
    np.uint32,
)

H, W = 8, 16
img_data = np.zeros((H, W, 4))
for i in range(H):
    for j in range(W):
        img_data[i, j, 0] = (i + 0.5) / H
        img_data[i, j, 1] = (j + 0.5) / W
        img_data[i, j, 3] = 1
img_data = (img_data * 255.0).astype(np.uint8)

class CustomViewer(Viewer):
    @hook
    def on_gui(self):
        if imgui.begin("YO")[0]:
            if implot.begin_plot("Plot", flags=implot.PlotFlags.NO_FRAME):
                implot.plot_line("Line", data[:, 0], data[:, 1][::-1],
                                 marker=Marker.SQUARE,
                                 marker_size=32,
                                 marker_fill_color=Vec4(1.0, 1.0, 0.0, 0.5),
                                 line_colors=colors,
                                 marker_line_colors=colors[::-1]
                                )
                implot.end_plot()
            if implot.begin_plot("Image", flags=implot.PlotFlags.NO_FRAME):
                implot.plot_image("RGB", texture, 0, 0, W, H)
                implot.plot_bubbles("Features",
                                    np.array([15, 6, 9], np.float32),
                                    np.array([1, 3, 2], np.float32),
                                    np.array([0.2, 0.5, 0.1],np.float32),
                                    line_colors=colors,
                                    fill_colors=colors,
                                    fill_alpha=0.5,
                                    )
                implot.plot_text("Text", 7, 5)
                implot.plot_dummy("Dummy")
                implot.end_plot()
            if implot.begin_plot("Heatmap", flags=implot.PlotFlags.NO_FRAME):
                implot.push_colormap(implot.Colormap.RD_BU)
                d = img_data[:, :, 0].astype(np.uint16)
                implot.plot_heatmap("RGB", d, label_fmt="%3.0f")
                implot.pop_colormap()
                implot.end_plot()
        imgui.end()


viewer = CustomViewer(
    config=Config(
        preferred_frames_in_flight=3,
        gui=GuiConfig(
            stats=True,
        ),
    ),
)

r = viewer.renderer
img = Image.from_data(r.device, img_data, ImageLayout.SHADER_READ_ONLY_OPTIMAL, W, H, Format.R8G8B8A8_UNORM, ImageUsageFlags.SAMPLED | ImageUsageFlags.TRANSFER_DST, AllocType.DEVICE)
descriptor_layout, descriptor_pool, descriptor_set = create_descriptor_layout_pool_and_set(
    r.device,
    [
        DescriptorSetBinding(1, DescriptorType.COMBINED_IMAGE_SAMPLER, stage_flags=Stage.FRAGMENT),
    ],
    r.num_frames_in_flight,
)
descriptor_set.write_combined_image_sampler(img, ImageLayout.SHADER_READ_ONLY_OPTIMAL, r.linear_sampler, 0)
texture = imgui.Texture(descriptor_set)


viewer.run()
