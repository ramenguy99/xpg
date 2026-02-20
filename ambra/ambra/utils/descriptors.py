# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from typing import List, Optional, Tuple

from pyxpg import (
    DescriptorPool,
    DescriptorPoolCreateFlags,
    DescriptorPoolSize,
    DescriptorSet,
    DescriptorSetBinding,
    DescriptorSetLayout,
    DescriptorSetLayoutCreateFlags,
    Device,
)

from .ring_buffer import RingBuffer


def create_descriptor_layout_pool_and_set(
    device: Device,
    bindings: List[DescriptorSetBinding],
    variable_count: int = 0,
    layout_flags: DescriptorSetLayoutCreateFlags = DescriptorSetLayoutCreateFlags.NONE,
    pool_flags: DescriptorPoolCreateFlags = DescriptorPoolCreateFlags.NONE,
    name: Optional[str] = None,
) -> Tuple[DescriptorSetLayout, DescriptorPool, DescriptorSet]:
    layout = DescriptorSetLayout(device, bindings, layout_flags, name=name)
    pool = DescriptorPool(device, [DescriptorPoolSize(b.count, b.type) for b in bindings], 1, pool_flags, name=name)
    set = pool.allocate_descriptor_set(layout, variable_count, name=name)
    return layout, pool, set


def create_descriptor_layout_pool_and_sets(
    device: Device,
    bindings: List[DescriptorSetBinding],
    count: int,
    variable_count: int = 0,
    layout_flags: DescriptorSetLayoutCreateFlags = DescriptorSetLayoutCreateFlags.NONE,
    pool_flags: DescriptorPoolCreateFlags = DescriptorPoolCreateFlags.NONE,
    name: Optional[str] = None,
) -> Tuple[DescriptorSetLayout, DescriptorPool, List[DescriptorSet]]:
    layout = DescriptorSetLayout(device, bindings, layout_flags, name=name)
    pool = DescriptorPool(
        device, [DescriptorPoolSize(b.count * count, b.type) for b in bindings], count, pool_flags, name=name
    )
    sets: List[DescriptorSet] = [pool.allocate_descriptor_set(layout, variable_count, name=name) for _ in range(count)]
    return layout, pool, sets


def create_descriptor_layout_pool_and_sets_ringbuffer(
    device: Device,
    bindings: List[DescriptorSetBinding],
    count: int,
    variable_count: int = 0,
    layout_flags: DescriptorSetLayoutCreateFlags = DescriptorSetLayoutCreateFlags.NONE,
    pool_flags: DescriptorPoolCreateFlags = DescriptorPoolCreateFlags.NONE,
    name: Optional[str] = None,
) -> Tuple[DescriptorSetLayout, DescriptorPool, RingBuffer[DescriptorSet]]:
    layout, pool, sets = create_descriptor_layout_pool_and_sets(
        device, bindings, count, variable_count, layout_flags, pool_flags, name
    )
    return layout, pool, RingBuffer(sets)


def create_descriptor_pool_and_sets(
    device: Device,
    layout: DescriptorSetLayout,
    count: int,
    variable_count: int = 0,
    pool_flags: DescriptorPoolCreateFlags = DescriptorPoolCreateFlags.NONE,
    name: Optional[str] = None,
) -> Tuple[DescriptorPool, List[DescriptorSet]]:
    pool = DescriptorPool(
        device, [DescriptorPoolSize(b.count * count, b.type) for b in layout.bindings], count, pool_flags, name=name
    )
    sets: List[DescriptorSet] = [pool.allocate_descriptor_set(layout, variable_count, name=name) for _ in range(count)]
    return pool, sets


def create_descriptor_pool_and_sets_ringbuffer(
    device: Device,
    layout: DescriptorSetLayout,
    count: int,
    variable_count: int = 0,
    pool_flags: DescriptorPoolCreateFlags = DescriptorPoolCreateFlags.NONE,
    name: Optional[str] = None,
) -> Tuple[DescriptorPool, RingBuffer[DescriptorSet]]:
    pool, sets = create_descriptor_pool_and_sets(device, layout, count, variable_count, pool_flags, name)
    return pool, RingBuffer(sets)
