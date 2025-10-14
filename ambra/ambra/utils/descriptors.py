# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from typing import List, Optional, Tuple

from pyxpg import (
    Context,
    DescriptorPool,
    DescriptorPoolCreateFlags,
    DescriptorPoolSize,
    DescriptorSet,
    DescriptorSetBinding,
    DescriptorSetLayout,
    DescriptorSetLayoutCreateFlags,
)

from .ring_buffer import RingBuffer


def create_descriptor_layout_pool_and_set(
    ctx: Context,
    bindings: List[DescriptorSetBinding],
    layout_flags: DescriptorSetLayoutCreateFlags = DescriptorSetLayoutCreateFlags.NONE,
    pool_flags: DescriptorPoolCreateFlags = DescriptorPoolCreateFlags.NONE,
    name: Optional[str] = None,
) -> Tuple[DescriptorSetLayout, DescriptorPool, DescriptorSet]:
    layout = DescriptorSetLayout(ctx, bindings, layout_flags, name=name)
    pool = DescriptorPool(ctx, [DescriptorPoolSize(b.count, b.type) for b in bindings], 1, pool_flags, name=name)
    set = pool.allocate_descriptor_set(layout, name=name)
    return layout, pool, set


def create_descriptor_layout_pool_and_sets(
    ctx: Context,
    bindings: List[DescriptorSetBinding],
    count: int,
    layout_flags: DescriptorSetLayoutCreateFlags = DescriptorSetLayoutCreateFlags.NONE,
    pool_flags: DescriptorPoolCreateFlags = DescriptorPoolCreateFlags.NONE,
    name: Optional[str] = None,
) -> Tuple[DescriptorSetLayout, DescriptorPool, List[DescriptorSet]]:
    layout = DescriptorSetLayout(ctx, bindings, layout_flags, name=name)
    pool = DescriptorPool(
        ctx, [DescriptorPoolSize(b.count * count, b.type) for b in bindings], count, pool_flags, name=name
    )
    sets: List[DescriptorSet] = [pool.allocate_descriptor_set(layout, name=name) for _ in range(count)]
    return layout, pool, sets


def create_descriptor_layout_pool_and_sets_ringbuffer(
    ctx: Context,
    bindings: List[DescriptorSetBinding],
    count: int,
    layout_flags: DescriptorSetLayoutCreateFlags = DescriptorSetLayoutCreateFlags.NONE,
    pool_flags: DescriptorPoolCreateFlags = DescriptorPoolCreateFlags.NONE,
    name: Optional[str] = None,
) -> Tuple[DescriptorSetLayout, DescriptorPool, RingBuffer[DescriptorSet]]:
    layout, pool, sets = create_descriptor_layout_pool_and_sets(ctx, bindings, count, layout_flags, pool_flags, name)
    return layout, pool, RingBuffer(sets)


def create_descriptor_pool_and_sets(
    ctx: Context,
    layout: DescriptorSetLayout,
    count: int,
    pool_flags: DescriptorPoolCreateFlags = DescriptorPoolCreateFlags.NONE,
    name: Optional[str] = None,
) -> Tuple[DescriptorPool, List[DescriptorSet]]:
    pool = DescriptorPool(
        ctx, [DescriptorPoolSize(b.count * count, b.type) for b in layout.bindings], count, pool_flags, name=name
    )
    sets: List[DescriptorSet] = [pool.allocate_descriptor_set(layout, name=name) for _ in range(count)]
    return pool, sets


def create_descriptor_pool_and_sets_ringbuffer(
    ctx: Context,
    layout: DescriptorSetLayout,
    count: int,
    pool_flags: DescriptorPoolCreateFlags = DescriptorPoolCreateFlags.NONE,
    name: Optional[str] = None,
) -> Tuple[DescriptorPool, RingBuffer[DescriptorSet]]:
    pool, sets = create_descriptor_pool_and_sets(ctx, layout, count, pool_flags, name)
    return pool, RingBuffer(sets)
