from typing import List
from time import perf_counter_ns
from enum import Enum, auto

from pyxpg import *
import numpy as np

ctx = Context(
    required_features=DeviceFeatures.SYNCHRONIZATION_2,
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
# - we can add support for memoryview parameters, but there is no automatic conversion (need wrap in memoryview())
# - also not sure what's the best way to create memory view objects for the outside, likely need to hold
#   a reference to the data in the current object. See how it works for arrays.
# - best would be to accept any buffer-like object as paramater, converting to contiguous only if necessary
#   and always return new contiguous memory views that live as long as the object lives.
#
# Upload perf:
# NVIDIA RTX 3060 Mobile
# -> First upload (likely hitting fixed costs like page table population)
#    - Host memory is always fastest   (6-9 GB /s)
#    - Bar memory is good < 32 MB/s    (5-8 GB /s)
#    - Bar memory is slow >= 32 MB/s    (2-3 GB /s)
#    - Gfx and transer are ok >= 8 MB (3-5 GB/s)
# -> Repeated upload (loop until 4GiB copied)
#    - Host Memory:    9 GB/s (when out of cache)
#    - Bar Memory:     9 GB/s (no caching)
#    - Gfx and transfer are best >=32 MB/s and <= 512 MB (12-18 GB/s), slows down at 1GiB+ (11 GB/s)
# Intel 11th gen UHD Graphics
#    - Host visible memory is (8-9 GB/s)
#    - Host visible write combining is slightly faster (9-10 GB/s) -> always prefer this
#    - Device memory + uplaod is slower (2-3 GB/s)
#    - No transfer queue
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
# - HOST_WRITE_COMBINING
# - DEVICE
# - DEVICE_MAPPED_WITH_FALLBACK
# - DEVICE_DEDICATED
#
# Uses:
# - with_data() -> use device visible memory for single alloc, fallback to copy (must add TRANSFER_DEST to prepare for fallback)
# - Staging / Readback buffers (HOST) -> could be skipped on unified?
# - Buffer creation with_data (DEVICE + bestperf heuristic)
# - Image creation with_data (DEVICE + sync upload always)
# - Uploads always go through API:
#   - If <= THRESHOLD and mapped: memcpy
#   - If >  THRESHOLD: copy command
#
# Helpers:
# - Transparent mapped buffer that is either DEVICE_MAPPED or HOST and is optionally uploaded before use -> useful for constants
#
# Integrated graphics:
# - Need to handle case were only 256 MB or no memory is DEVICE_LOCAL -> VMA can probably help with this with PREFER DEVICE style stuff
# - Need special APIs for unified memory? Skip uploads all together and only use mapped memory?
#   - This requires awarness also at the application level to know it can rely on mapping certain things that normally would not make sense to map (e.g. large buffers)
#
# Improvements (for later):
# - expose host_image_copy for more image copy options

copy_cmd = CommandBuffer(ctx, ctx.transfer_queue_family_index)
copy_queue = ctx.transfer_queue
fence = Fence(ctx)

TARGET_SIZE = 4 << 30

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

    for i in range(10, 32):
        N = 1 << i
        arr = np.ones(N, np.uint8)

        alloc_type = AllocType.HOST
        if method == Method.BAR or method == Method.HOST:
            buf = Buffer.from_data(ctx, arr, BufferUsageFlags.STORAGE, AllocType.HOST if method == Method.HOST else AllocType.DEVICE_MAPPED)
            # print(f"{buf.alloc}")
        else:
            staging = Buffer.from_data(ctx, arr, BufferUsageFlags.TRANSFER_SRC, AllocType.HOST)
            gpu = Buffer(ctx, len(arr), BufferUsageFlags.TRANSFER_DST | BufferUsageFlags.STORAGE, AllocType.DEVICE)
            with ctx.sync_commands() as cmd:
                cmd.copy_buffer(staging, gpu)

        total_begin = perf_counter_ns()

        # BAR = True
        ITERS = min(TARGET_SIZE // N, 1000)
        for _ in range(ITERS):
            if method == Method.BAR or method == Method.HOST:
                buf.data[:] = arr.data
            elif method == Method.GFX_COPY:
                    with ctx.sync_commands() as cmd:
                        cmd.copy_buffer(staging, gpu)
            elif method == Method.TRANSFER_COPY:
                    with copy_cmd:
                        copy_cmd.copy_buffer(staging, gpu)
                    copy_queue.submit(copy_cmd, fence=fence)
                    fence.wait_and_reset()
            else:
                assert False

        total_end = perf_counter_ns()
        delta = (total_end - total_begin) * 1e-9 / ITERS
        print(f"{N >> 20:4d} MB: {delta * 1e3:10.3f}ms | {N / 2**20  / delta:10.3f} MB/s")
        
        if method == Method.BAR or method == Method.HOST:
            buf.destroy()
        else:
            staging.destroy()
            gpu.destroy()


