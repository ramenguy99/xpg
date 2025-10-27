import numpy as np
from pyxpg import AllocType, Buffer, BufferUsageFlags, MemoryUsage

from ambra.config import Config
from ambra.gpu_sorting import GpuSortingPipeline, SortDataType, SortOptions
from ambra.viewer import Viewer

v = Viewer(
    config=Config(
        window=False,
    )
)

# Prepare pipleine
sort = GpuSortingPipeline(v.renderer, SortOptions(key_type=SortDataType.UINT32, payload_type=SortDataType.UINT32))

# Create random keys
np.random.seed(42)
N = 65535 * 32 + 17300
keys_array = np.random.randint(0, 0xFFFFFFFF, N, np.uint32)

# Alocate buffers
keys = Buffer.from_data(
    v.ctx,
    keys_array,
    BufferUsageFlags.STORAGE | BufferUsageFlags.TRANSFER_DST | BufferUsageFlags.TRANSFER_SRC,
    AllocType.DEVICE_MAPPED_WITH_FALLBACK,
    name="keys",
)
keys_alt = Buffer(v.ctx, keys.size, BufferUsageFlags.STORAGE, AllocType.DEVICE, name="keys-alt")

payload = Buffer.from_data(
    v.ctx,
    np.arange(N, dtype=np.uint32),
    BufferUsageFlags.STORAGE | BufferUsageFlags.TRANSFER_DST | BufferUsageFlags.TRANSFER_SRC,
    AllocType.DEVICE_MAPPED_WITH_FALLBACK,
    name="payload",
)
payload_alt = Buffer(v.ctx, payload.size, BufferUsageFlags.STORAGE, AllocType.DEVICE, name="payload-alt")

readback_keys = Buffer(
    v.ctx, keys.size, BufferUsageFlags.STORAGE | BufferUsageFlags.TRANSFER_DST, AllocType.HOST, name="readback-keys"
)
readback_payload = Buffer(
    v.ctx,
    payload.size,
    BufferUsageFlags.STORAGE | BufferUsageFlags.TRANSFER_DST,
    AllocType.HOST,
    name="readback-payload",
)

# Sort and readback
with v.ctx.sync_commands() as cmd:
    sort.run(v.renderer, cmd, N, keys, keys_alt, payload, payload_alt)
    cmd.memory_barrier(MemoryUsage.COMPUTE_SHADER, MemoryUsage.TRANSFER_SRC)
    cmd.copy_buffer(keys, readback_keys)
    cmd.copy_buffer(payload, readback_payload)

# Check results
readback_keys_array = np.frombuffer(readback_keys.data, np.uint32)
readback_payload_array = np.frombuffer(readback_payload.data, np.uint32)

indices_array = np.argsort(keys_array, kind="stable").astype(np.uint32)
sorted_keys = keys_array[indices_array]

assert np.array_equal(readback_keys_array, sorted_keys)
assert np.array_equal(readback_payload_array, indices_array)
