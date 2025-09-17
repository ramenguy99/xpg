from typing import List, Optional, Tuple

from pyxpg import (
    Context,
    DescriptorPool,
    DescriptorPoolSize,
    DescriptorSet,
    DescriptorSetBinding,
    DescriptorSetLayout,
)

from .ring_buffer import RingBuffer


def create_descriptor_layout_pool_and_set(
    ctx: Context, bindings: List[DescriptorSetBinding], name: Optional[str] = None
) -> Tuple[DescriptorSetLayout, DescriptorPool, DescriptorSet]:
    layout = DescriptorSetLayout(ctx, bindings, name=name)
    pool = DescriptorPool(ctx, [DescriptorPoolSize(b.count, b.type) for b in bindings], 1, name=name)
    set = pool.allocate_descriptor_set(layout, name=name)
    return layout, pool, set


def create_descriptor_layout_pool_and_sets(
    ctx: Context, bindings: List[DescriptorSetBinding], count: int, name: Optional[str] = None
) -> Tuple[DescriptorSetLayout, DescriptorPool, List[DescriptorSet]]:
    layout = DescriptorSetLayout(ctx, bindings, name=name)
    pool = DescriptorPool(ctx, [DescriptorPoolSize(b.count * count, b.type) for b in bindings], count, name=name)
    sets: List[DescriptorSet] = [pool.allocate_descriptor_set(layout, name=name) for _ in range(count)]
    return layout, pool, sets


def create_descriptor_layout_pool_and_sets_ringbuffer(
    ctx: Context, bindings: List[DescriptorSetBinding], count: int, name: Optional[str] = None
) -> Tuple[DescriptorSetLayout, DescriptorPool, RingBuffer[DescriptorSet]]:
    layout, pool, sets = create_descriptor_layout_pool_and_sets(ctx, bindings, count, name)
    return layout, pool, RingBuffer(sets)


def create_descriptor_pool_and_sets(
    ctx: Context, layout: DescriptorSetLayout, count: int, name: Optional[str] = None
) -> Tuple[DescriptorPool, RingBuffer[DescriptorSet]]:
    pool = DescriptorPool(
        ctx, [DescriptorPoolSize(b.count * count, b.type) for b in layout.bindings], count, name=name
    )
    sets: List[DescriptorSet] = [pool.allocate_descriptor_set(layout, name=name) for _ in range(count)]
    return pool, sets


def create_descriptor_pool_and_sets_ringbuffer(
    ctx: Context, bindings: List[DescriptorSetBinding], count: int, name: Optional[str] = None
) -> Tuple[DescriptorPool, RingBuffer[DescriptorSet]]:
    pool, sets = create_descriptor_pool_and_sets(ctx, bindings, count, name)
    return pool, RingBuffer(sets)
