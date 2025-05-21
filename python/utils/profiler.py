from contextlib import contextmanager
from dataclasses import dataclass
from queue import Queue
from typing import Optional, List
from time import perf_counter_ns

from pyxpg import Context, QueryPool, QueryType, CommandBuffer, PipelineStageFlags, DeviceFeatures


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

        self.pools = [QueryPool(ctx, QueryType.TIMESTAMP, max_gpu_zones * 2) for _ in range(num_frames)]
        if ctx.has_transfer_queue:
            self.transfer_pools = [QueryPool(ctx, QueryType.TIMESTAMP, max_gpu_zones * 2) for _ in range(num_frames)]
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
    