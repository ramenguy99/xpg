from contextlib import contextmanager
from dataclasses import dataclass
from queue import Queue
from typing import Optional, List
from time import perf_counter_ns
from hashlib import md5

from pyxpg import Context, QueryPool, QueryType, CommandBuffer, PipelineStageFlags, DeviceFeatures, imgui


@dataclass
class Zone:
    name: str
    start_idx: int
    end_idx: int = None
    start_time: int = None
    end_time: int = None
    depth: int = 0

@dataclass
class ProfilerFrame:
    frame: int
    frame_start_ts: int
    frame_start_gpu_ts: int
    zones: List[Zone]
    gpu_zones: List[Zone]
    gpu_transfer_zones: List[Zone]

class Profiler:
    def __init__(self, ctx: Context, num_frames: int, max_gpu_zones: int = 64):
        self.ctx = ctx
        self.num_frames = num_frames
        self.max_gpu_zones = max_gpu_zones

        self.pools = [QueryPool(ctx, QueryType.TIMESTAMP, max_gpu_zones * 2, name=f"profiler-query-pool-{i}") for i in range(num_frames)]
        if ctx.has_transfer_queue:
            self.transfer_pools = [QueryPool(ctx, QueryType.TIMESTAMP, max_gpu_zones * 2, name=f"profiler-query-pool-transfer-{i}") for i in range(num_frames)]
        else:
            self.transfer_pools = None

        self.current_cmd = None
        self.current_transfer_cmd = None
        self.current_query = 0
        self.current_cpu_zone = 0
        self.current_cpu_depth = 0
        self.current_cpu_depth = 0

        self.zones = []
        self.gpu_zones = []
        self.gpu_transfer_zones = []
        self.frame_index = -1
        self.total_frame_index = -1

        self.results = Queue()

    
    def frame(self, cmd: CommandBuffer) -> Optional[ProfilerFrame]:
        self.frame_index = (self.frame_index + 1) % len(self.pools)
        self.total_frame_index += 1

        res = None
        if self.total_frame_index >= self.num_frames:
            res: ProfilerFrame = self.results.get()
            if res.gpu_zones:
                timestamps = self.pools[self.frame_index].wait_results(0, len(res.gpu_zones * 2))
                for z in res.gpu_zones:
                    z.start_time = timestamps[z.start_idx]
                    z.end_time = timestamps[z.end_idx]

            if self.transfer_pools and res.gpu_transfer_zones:
                transfer_timestamps = self.transfer_pools[self.frame_index].wait_results(0, len(res.gpu_transfer_zones * 2))
                for z in res.gpu_transfer_zones:
                    z.start_time = transfer_timestamps[z.start_idx]
                    z.end_time = transfer_timestamps[z.end_idx]
            if res.frame_start_gpu_ts == 0 and len(res.gpu_zones) > 0:
                res.frame_start_gpu_ts = res.gpu_zones[0].start_time

        self.zones = []
        self.gpu_zones = []
        self.gpu_transfer_zones = []

        host_ts, device_ts = self.ctx.get_calibrated_timestamps()
        self.results.put(ProfilerFrame(self.total_frame_index, host_ts, device_ts, self.zones, self.gpu_zones, self.gpu_transfer_zones))

        self.current_query = 0
        self.current_cpu_zone = 0
        self.current_cmd = cmd
        self.current_cmd.reset_query_pool(self.pools[self.frame_index])

        self.current_cpu_depth = 0
        self.current_gpu_depth = 0
        self.current_gpu_transfer_depth = 0


        return res
        
    def transfer_frame(self, transfer_cmd: CommandBuffer):
        self.current_transfer_query = 0
        self.current_transfer_cmd = transfer_cmd

        if not self.ctx.device_features & DeviceFeatures.HOST_QUERY_RESET:
            raise RuntimeError("DeviceFeatures.HOST_QUERY_RESET must be enabled to profile the transfer queue")
        else:
            self.ctx.reset_query_pool(self.transfer_pools[self.frame_index])
    
    @contextmanager
    def zone(self, name):
        start = self.current_cpu_zone
        self.current_cpu_zone += 1

        zone = Zone(name, start, start_time=perf_counter_ns(), depth=self.current_cpu_depth)
        self.zones.append(zone)

        self.current_cpu_depth += 1
        try:
            yield
        finally:
            self.current_cpu_depth -= 1
            end = self.current_cpu_zone
            self.current_cpu_zone += 1

            zone.end_idx = end
            zone.end_time = perf_counter_ns()

    @contextmanager
    def gpu_zone(self, name):
        if len(self.gpu_zones) > self.max_gpu_zones:
            return

        start = self.current_query
        self.current_query += 1

        zone = Zone(name, start, depth=self.current_gpu_depth)
        self.gpu_zones.append(zone)

        self.current_cmd.write_timestamp(self.pools[self.frame_index], start, PipelineStageFlags.TOP_OF_PIPE)
        self.current_gpu_depth += 1
        try:
            yield
        finally:
            self.current_gpu_depth -= 1
            end = self.current_query
            self.current_query += 1

            zone.end_idx = end

            self.current_cmd.write_timestamp(self.pools[self.frame_index], end, PipelineStageFlags.BOTTOM_OF_PIPE)

    @contextmanager
    def gpu_transfer_zone(self, name):
        if len(self.gpu_transfer_zones) > self.max_gpu_zones:
            return

        start = self.current_transfer_query
        self.current_transfer_query += 1

        zone = Zone(name, start, depth=self.current_gpu_transfer_depth)
        self.gpu_transfer_zones.append(zone)

        self.current_transfer_cmd.write_timestamp(self.transfer_pools[self.frame_index], start, PipelineStageFlags.TOP_OF_PIPE)
        self.current_gpu_transfer_depth += 1
        try:
            yield
        finally:
            self.current_gpu_transfer_depth -= 1
            end = self.current_transfer_query
            self.current_transfer_query += 1

            zone.end_idx = end

            self.current_transfer_cmd.write_timestamp(self.transfer_pools[self.frame_index], end, PipelineStageFlags.BOTTOM_OF_PIPE)
    


def gui_profiler_list(prof: ProfilerFrame, dt: float, timestamp_period_ns: float):
    if imgui.begin("Profiler - list")[0]:
        imgui.text(f"dt: {dt * 1000:.3f}ms")
        to_ms = timestamp_period_ns * 1e-6
        imgui.separator_text("CPU")
        if prof:
            for z in prof.zones:
                imgui.text(f"{z.name}: {(z.end_time - z.start_time) * 1e-6:.3f}ms")
        imgui.separator_text("GFX")
        if prof:
            for z in prof.gpu_zones:
                imgui.text(f"{z.name}: {(z.end_time - z.start_time) * to_ms:.3f}ms")
        imgui.separator_text("Transfer")
        if prof:
            for z in prof.gpu_transfer_zones:
                imgui.text(f"{z.name}: {(z.end_time - z.start_time) * to_ms:.3f}ms")
    imgui.end()


hovered_frame = -1
def gui_profiler_graph(profiler_results: List[ProfilerFrame], timestamp_period_ns: float):
    global hovered_frame

    if imgui.begin("Profiler - graph")[0] and profiler_results:
        pos = imgui.get_mouse_pos()
        expected_width = 1000 // 5
        expected_length = 20

        dl = imgui.get_window_draw_list()
        HEIGHT = 30

        start_ts = profiler_results[0].frame_start_ts
        start_gpu_ts = profiler_results[0].frame_start_gpu_ts


        hovered_something = False
        for i, name in enumerate(["CPU", "GPU", "GPU Transfer"]):
            imgui.separator_text(name)
            start = imgui.get_cursor_screen_pos()
            for prof in profiler_results:
                zones = [prof.zones, prof.gpu_zones, prof.gpu_transfer_zones][i]
                if i == 0:
                    to_ms = 1e-6
                    min_ts = start_ts
                else:
                    to_ms = 1e-6 * timestamp_period_ns
                    min_ts = start_gpu_ts
                norm = to_ms / expected_length * expected_width

                if zones:
                    c = imgui.Vec2(start.x, start.y)

                    # max_ts = min([z.end_time for z in prof.zones])

                    for z in zones:
                        # Replace with something reasonable
                        r, g, b = md5(z.name.encode(), usedforsecurity=False).digest()[:3]

                        s = (z.start_time - min_ts) * norm
                        e = (z.end_time - min_ts) * norm
                        e = max(e, s+1)


                        x0 = c.x + s
                        x1 = c.x + e
                        y0 = c.y + HEIGHT * z.depth
                        y1 = c.y + HEIGHT * (z.depth + 1)

                        outline_color = 0xFFCCCCCC
                        if pos.x >= x0 and pos.x < x1 and pos.y >= y0 and pos.y < y1:
                            hovered_frame = prof.frame
                            hovered_something = True
                            imgui.begin_tooltip()
                            imgui.text(f"Frame: {prof.frame}")
                            imgui.text(f"{z.name}")
                            imgui.text(f"Duration: {(z.end_time - z.start_time) * to_ms:.3f}ms")
                            imgui.end_tooltip()
                            outline_color = 0xFFFFFFFF
                        
                        # e = max(e, s+20)
                        if prof.frame == hovered_frame:
                            dl.add_rect((x0-1, y0-1), (x1+1, y1+1), outline_color, thickness=2)
                            r = min(r + 40, 255)
                            g = min(g + 40, 255)
                            b = min(b + 40, 255)

                        dl.add_rect_filled((x0, y0), (x1, y1), 0xFF000000 | (b << 16) | (g << 8) | r )

            imgui.set_cursor_screen_pos((c.x, c.y + HEIGHT * 3.5))
        if not hovered_something:
            hovered_frame = -1
    imgui.end()

