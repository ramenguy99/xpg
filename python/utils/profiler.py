from contextlib import contextmanager
from dataclasses import dataclass

from pyxpg import Context, QueryPool, QueryType, CommandBuffer, PipelineStageFlags
from queue import Queue

from typing import Optional, List

@dataclass
class Zone:
    name: str
    start_idx: int
    end_idx: int = None
    start_time: int = None
    end_time: int = None

@dataclass
class Result:
    frame: int
    zones: List[Zone]
    transfer_zones: List[Zone]

class GpuProfiler:
    def __init__(self, ctx: Context, num_frames: int, max_zones: int = 64):
        self.ctx = ctx
        self.num_frames = num_frames
        self.max_zones = max_zones

        self.pools = [QueryPool(ctx, QueryType.TIMESTAMP, max_zones * 2) for _ in range(num_frames)]
        if ctx.has_transfer_queue:
            self.transfer_pools = [QueryPool(ctx, QueryType.TIMESTAMP, max_zones * 2) for _ in range(num_frames)]
        else:
            self.transfer_pools = None

        self.current_cmd = None
        self.current_transfer_cmd = None

        self.zones = []
        self.transfer_zones = []
        self.frame_index = -1
        self.total_frame_index = -1

        self.results = Queue()

    
    def frame(self, cmd: CommandBuffer) -> Optional[Result]:
        self.frame_index = (self.frame_index + 1) % len(self.pools)
        self.total_frame_index += 1

        res = None
        if self.total_frame_index >= self.num_frames:
            res: Result = self.results.get()
            if res.zones:
                timestamps = self.pools[self.frame_index].wait_results(0, len(res.zones * 2))
                for z in res.zones:
                    z.start_time = timestamps[z.start_idx]
                    z.end_time = timestamps[z.end_idx]

            if self.transfer_pools and res.transfer_zones:
                transfer_timestamps = self.transfer_pools[self.frame_index].wait_results(0, len(res.transfer_zones * 2))
                for z in res.transfer_zones:
                    z.start_time = transfer_timestamps[z.start_idx]
                    z.end_time = transfer_timestamps[z.end_idx]

        self.zones = []
        self.transfer_zones = []
        self.results.put(Result(self.frame_index, self.zones, self.transfer_zones))

        self.current_query = 0
        self.current_cmd = cmd
        self.current_cmd.reset_query_pool(self.pools[self.frame_index])

        return res
        
    def transfer(self, transfer_cmd: CommandBuffer):
        self.current_transfer_query = 0
        self.current_transfer_cmd = transfer_cmd

        with self.ctx.sync_commands() as cmd:
            cmd.reset_query_pool(self.transfer_pools[self.frame_index])
    
    @contextmanager
    def zone(self, name, cmd: CommandBuffer = None):
        if len(self.zones) > self.max_zones:
            return

        start = self.current_query
        self.current_query += 1

        zone = Zone(name, start)
        self.zones.append(zone)

        cmd = cmd if cmd else self.current_cmd
        cmd.write_timestamp(self.pools[self.frame_index], start, PipelineStageFlags.TOP_OF_PIPE)
        try:
            yield
        finally:
            end = self.current_query
            self.current_query += 1

            zone.end_idx = end

            cmd.write_timestamp(self.pools[self.frame_index], end, PipelineStageFlags.BOTTOM_OF_PIPE)

    @contextmanager
    def transfer_zone(self, name):
        if len(self.transfer_zones) > self.max_zones:
            return

        start = self.current_transfer_query
        self.current_transfer_query += 1

        zone = Zone(name, start)
        self.transfer_zones.append(zone)

        self.current_transfer_cmd.write_timestamp(self.transfer_pools[self.frame_index], start, PipelineStageFlags.TOP_OF_PIPE)
        try:
            yield
        finally:
            end = self.current_transfer_query
            self.current_transfer_query += 1

            zone.end_idx = end

            self.current_cmd.write_timestamp(self.transfer_pools[self.frame_index], end, PipelineStageFlags.BOTTOM_OF_PIPE)
    