import numpy as np
from pyxpg import AllocType, Buffer, BufferUsageFlags

from ambra.config import Config
from ambra.gpu_sorting import GpuSortingPipeline, SortDataType, SortOptions
from ambra.viewer import Viewer

v = Viewer(
    config=Config(
        window=False,
        enable_gpu_based_validation=False,
    )
)

# Prepare pipleine
sort = GpuSortingPipeline(v.renderer, SortOptions(key_type=SortDataType.UINT32))

np.random.seed(42)

N = 4
keys_array = np.random.randint(0, 0xFF, N, np.uint32)

# Alocate buffers
keys = Buffer.from_data(
    v.ctx,
    keys_array,
    BufferUsageFlags.STORAGE | BufferUsageFlags.TRANSFER_DST | BufferUsageFlags.TRANSFER_SRC,
    AllocType.DEVICE_MAPPED_WITH_FALLBACK,
    name="keys",
)
keys_alt = Buffer(v.ctx, keys.size, BufferUsageFlags.STORAGE, AllocType.DEVICE, name="keys-alt")
readback = Buffer(
    v.ctx, keys.size, BufferUsageFlags.STORAGE | BufferUsageFlags.TRANSFER_DST, AllocType.HOST, name="readback"
)

# Sort
with v.ctx.sync_commands() as cmd:
    sort.run(v.renderer, cmd, N, keys, keys_alt, v.renderer.zero_buffer, v.renderer.zero_buffer)
    cmd.copy_buffer(keys, readback)

readback_array = np.frombuffer(readback.data, np.uint32)
print(keys_array.shape, readback_array.shape)
print(keys_array)
print(readback_array)
