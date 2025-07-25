from pyxpg import Context, Buffer, BufferUsageFlags, AllocType, CommandBuffer, MemoryUsage, DescriptorSet, DescriptorSetEntry, DescriptorType
from typing import Optional, List
from dataclasses import dataclass

from .ring_buffer import RingBuffer

class UploadableBuffer(Buffer):
    def __init__(self, ctx: Context, size: int, usage_flags: BufferUsageFlags, name: Optional[str] = None):
        super().__init__(ctx, size, usage_flags | BufferUsageFlags.TRANSFER_DST, AllocType.DEVICE_MAPPED_WITH_FALLBACK, name)
        if not self.is_mapped:
            if name is not None:
                name = f"{name} - staging"
            self._staging = Buffer(ctx, size, BufferUsageFlags.TRANSFER_SRC, AllocType.HOST_WRITE_COMBINING, name)

    @classmethod
    def from_data(cls, ctx: Context, data: memoryview, usage_flags: BufferUsageFlags, name: Optional[str] = None):
        buf = cls(ctx, len(data), usage_flags, name=name)
        buf.upload_sync(data)
        return buf

    def upload(self, cmd: CommandBuffer, usage: MemoryUsage, data: memoryview, offset: int = 0):
        if self.is_mapped:
            # print(self.data.c_contiguous, data.c_contiguous )
            # print(self.data.shape       , data.shape        )
            # print(self.data.readonly    , data.readonly     )
            # print(self.data.format    , data.format     )
            self.data[offset:offset+len(data)] = data
        else:
            self._staging.data[offset:offset+len(data)] = data
            cmd.copy_buffer_range(self._staging, self, len(data), src_offset=offset, dst_offset=offset)
            if usage != MemoryUsage.NONE:
                cmd.memory_barrier(MemoryUsage.TRANSFER_WRITE, usage)

    def upload_sync(self, data: memoryview, offset: int = 0):
        with self.ctx.sync_commands() as cmd:
            self.upload(cmd, MemoryUsage.NONE, data, offset)

@dataclass
class UniformBlockAllocation:
    descriptor_set: DescriptorSet
    buffer: UploadableBuffer
    offset: int
    size: int

    def upload(self, cmd: CommandBuffer, data: memoryview):
        if self.size < len(data):
            raise IndexError("data is larger than buffer allocation")
        self.buffer.upload(cmd, MemoryUsage.ANY_SHADER_UNIFORM_READ, data, self.offset)

@dataclass
class UniformBlock:
    descriptor_sets: RingBuffer[DescriptorSet]
    buffers: RingBuffer[Buffer]
    size: int
    used: int = 0

    def alloc(self, size: int, alignment: int) -> UniformBlockAllocation:
        assert self.used + size < self.size
        alloc = UniformBlockAllocation(self.descriptor_sets.get_current(),  self.buffers.get_current(), self.used, size)
        self.used += (size + alignment - 1) & ~(alignment - 1)
        return alloc

    def advance(self):
        self.descriptor_sets.advance()
        self.buffers.advance()
        self.used = 0

class UniformPool:
    def __init__(self, ctx: Context, num_frames: int, block_size: int):
        self.ctx = ctx
        self.num_frames = num_frames
        self.block_size = block_size
        self.blocks: List[UniformBlock] = []
        self.alignment = max(16, self.ctx.device_properties.limits.min_uniform_buffer_offset_alignment)
        self.max_uniform_buffer_range = self.ctx.device_properties.limits.max_uniform_buffer_range

        # Warmup first block
        self._alloc_block(block_size)

        # Grab a descriptor set for pipeline layouts
        self.descriptor_set = self.blocks[0].descriptor_sets.get_current()

    def _alloc_block(self, min_size: int):
        size = max(min_size, self.block_size)
        block_idx = len(self.blocks)
        block = UniformBlock(
            descriptor_sets = RingBuffer(self.num_frames, DescriptorSet, self.ctx, [
                DescriptorSetEntry(1, DescriptorType.UNIFORM_BUFFER_DYNAMIC),
            ], name=f"set - uniform pool block {block_idx}"),
            buffers = RingBuffer(self.num_frames, UploadableBuffer, self.ctx, size, BufferUsageFlags.UNIFORM, name=f"set - uniform pool block {block_idx}"),
            size = size,
        )
        for s, b in zip(block.descriptor_sets.items, block.buffers.items):
            s.write_buffer(b, DescriptorType.UNIFORM_BUFFER_DYNAMIC, 0, size=self.max_uniform_buffer_range)

        # Sync ringbuffer index, not necessary but makes for easier debugging
        if self.blocks:
            index = self.blocks[0].buffers.index
            block.buffers.set(index)
            block.descriptor_sets.set(index)

        self.blocks.append(block)
        return block

    def alloc(self, size: int) -> UniformBlockAllocation:
        if size > self.max_uniform_buffer_range:
            raise RuntimeError(f"allocation size ({size}) is larger than device limit ({self.max_uniform_buffer_range})")

        for b in self.blocks:
            if b.used + size < b.size:
                return b.alloc(size, self.alignment)

        # No space left, alloca a new block
        return self._alloc_block(size).alloc(size, self.alignment)

    def advance(self):
        for b in self.blocks:
            b.advance()

