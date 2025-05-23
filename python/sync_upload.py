from typing import List
from time import perf_counter_ns
from enum import Enum, auto

from pyxpg import *
import numpy as np


ctx = Context(
    device_features=DeviceFeatures.SYNCHRONIZATION_2,
    enable_validation_layer=True,
    enable_synchronization_validation=True,
)

# Findings:
# - a.tobytes()  copies the numpy array -> as per docs
# - a.data       does not copy
# - nb::bytes    does not copy
# - nb::bytes&   also does not copy
#
# Nanobind:
# - we can add support for memoryview parameters, but there is no automatic conversion (neet wrap in memoryview())
# - also not sure what's the best way to create memory view objects for the outside, likely need to hold
#   a reference to the data in the current object. See how it works for arrays.
#
# Upload perf:
# NVIDIA RTX 3060 Mobile
# - Host memory is always fastest   (6-8 GB /s)
# - Bar memory is good < 32 MB/s    (3-5 GB /s)
# - Bar memory is slow > 32 MB/s    (1 GB /s)
# - Gfx and transer are ok >= 16 MB (2 GB/s)
#
# Best perf:
#  if unified memory:
#       use DEVICE_MAPPED
#  else:
#       if DEVICE_MAPPED available:
#           if size < 32 MB:
#               use DEVICE_MAPPED
#           else:
#               use HOST + COPY
#       else:
#           use HOST + COPY
#
# Memory types exposed:
# - HOST
# - DEVICE
# - DEVICE_DEDICATED
# - DEVICE_PREFER_MAPPED
#
# Uses:
# - Buffer creation with_data (DEVICE + bestperf heuristic)
# - Image creation with_data (DEVICE + sync upload always)
# - Uploads always go through API:
#   - If <= THRESHOLD and mapped: memcpy
#   - If >  THRESHOLD: copy command
#
# Helpers:
# - Transparent mapped buffer that is either DEVICE_MAPPED or HOST and is optionally uploaded before use -> useful for constants
#
# Improvements (for later):
# - expose host_image_copy for more image copy options
# - Special APIs for unified memory? Skip uploads alltogether and only use mapped memory?
#   - This requires awarness also at the application level to know it can rely on mapping certain things

copy_cmd = CommandBuffer(ctx, ctx.transfer_queue_family_index)
copy_queue = ctx.transfer_queue
fence = Fence(ctx)

class Method(Enum):
    HOST = auto()
    BAR = auto()
    GFX_COPY = auto()
    TRANSFER_COPY = auto()

for method in [
    Method.HOST,
    Method.BAR,
    Method.GFX_COPY,
    Method.TRANSFER_COPY,
]:
    print(method)

    for i in range(10, 30):
        N = 1 << i
        data = np.ones(N, np.uint8)
        byt = data.tobytes()

        total_begin = perf_counter_ns()

        # BAR = True
        if method == Method.HOST:
            Buffer.from_data(ctx, byt, BufferUsageFlags.STORAGE, AllocType.HOST)
        elif method == Method.BAR:
            Buffer.from_data(ctx, byt, BufferUsageFlags.STORAGE, AllocType.DEVICE_MAPPED)
        else:
            begin = perf_counter_ns()
            host = Buffer.from_data(ctx, byt, BufferUsageFlags.TRANSFER_SRC, AllocType.HOST)
            end = perf_counter_ns()
            host_delta = (end - begin) * 1e-9

            begin = perf_counter_ns()
            gpu = Buffer(ctx, len(byt), BufferUsageFlags.TRANSFER_DST | BufferUsageFlags.STORAGE, AllocType.DEVICE)
            end = perf_counter_ns()
            alloc_delta = (end - begin) * 1e-9

            begin = perf_counter_ns()
            if method == Method.GFX_COPY:
                    with ctx.sync_commands() as cmd:
                        cmd.copy_buffer(host, gpu)
            elif method == Method.TRANSFER_COPY:
                    with copy_cmd:
                        copy_cmd.copy_buffer(host, gpu)
                    copy_queue.submit(copy_cmd, fence=fence)
                    fence.wait_and_reset()
            else:
                 assert False

            end = perf_counter_ns()
            copy_delta = (end - begin) * 1e-9
            # print(f"  Host: {host_delta * 1e3:10.3f}ms | {(N >> 20) / host_delta:10.3f} MB/s")
            # print(f"  Allc: {alloc_delta * 1e3:10.3f}ms | {(N >> 20) / alloc_delta:10.3f} MB/s")
            # print(f"  Copy: {copy_delta * 1e3:10.3f}ms | {(N >> 20) / copy_delta:10.3f} MB/s")

        total_end = perf_counter_ns()
        delta = (total_end - total_begin) * 1e-9
        print(f"{N >> 20:3d} MB: {delta * 1e3:10.3f}ms | {(N >> 20) / delta:10.3f} MB/s")


        # Buffer.from_data(ctx, data.data, BufferUsageFlags.STORAGE, AllocType.DEVICE_MAPPED)
        # Buffer.from_data_perf(ctx, memoryview(data.tobytes()), BufferUsageFlags.STORAGE, AllocType.DEVICE_MAPPED)
        # Buffer.from_data_value(ctx, byt, BufferUsageFlags.STORAGE, AllocType.DEVICE_MAPPED)


