import numpy as np
from pyxpg import AllocType, Buffer, BufferUsageFlags, MemoryUsage, PipelineStageFlags, QueryPool, QueryType

from ambra.config import Config
from ambra.gpu_sorting import GpuSortingPipeline, SortDataType, SortOptions
from ambra.viewer import Viewer

v = Viewer(
    config=Config(
        window=False,
    )
)

# Prepare pipleine
sort = GpuSortingPipeline(
    v.renderer,
    SortOptions(
        key_type=SortDataType.UINT32,
        payload_type=SortDataType.UINT32,
        unsafe_has_forward_thread_progress_guarantee=False,
    ),
)

# Create random keys
np.random.seed(42)
N = 1 << 27
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

BENCHMARK = True
if BENCHMARK:
    for i in range(27):
        n = 1 << i
        dt = 1000_000_000
        for _ in range(5):
            pool = QueryPool(v.ctx, QueryType.TIMESTAMP, 2)
            v.ctx.reset_query_pool(pool)
            with v.ctx.sync_commands() as cmd:
                cmd.write_timestamp(pool, 0, PipelineStageFlags.TOP_OF_PIPE)
                sort.run(v.renderer, cmd, n, keys, keys_alt, payload, payload_alt)
                cmd.memory_barrier(MemoryUsage.COMPUTE_SHADER, MemoryUsage.TRANSFER_SRC)
                cmd.write_timestamp(pool, 1, PipelineStageFlags.BOTTOM_OF_PIPE)
            res = pool.wait_results(0, 2)
            dt = min(dt, res[1] - res[0])
        dt_ns = dt * v.ctx.timestamp_period_ns
        print(f"{n:12}: {dt_ns * 1e-3:12.03f}us  {n / (dt_ns)} B/s")
else:
    readback_keys = Buffer(
        v.ctx,
        keys.size,
        BufferUsageFlags.STORAGE | BufferUsageFlags.TRANSFER_DST,
        AllocType.HOST,
        name="readback-keys",
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
        cmd.memory_barrier(MemoryUsage.TRANSFER_SRC, MemoryUsage.HOST_READ)

    # Check results
    readback_keys_array = np.frombuffer(readback_keys.data, np.uint32)
    readback_payload_array = np.frombuffer(readback_payload.data, np.uint32)

    indices_array = np.argsort(keys_array, kind="stable").astype(np.uint32)
    sorted_keys = keys_array[indices_array]

    assert np.array_equal(readback_keys_array, sorted_keys)
    assert np.array_equal(readback_payload_array, indices_array)
