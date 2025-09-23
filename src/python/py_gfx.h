// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/intrusive/counter.h>
#include <nanobind/intrusive/ref.h>

#include <xpg/gfx.h>

struct Context;

struct GfxObject: nanobind::intrusive_base {
    GfxObject() {}
    GfxObject(nanobind::ref<Context> ctx, bool owned, std::optional<nanobind::str> name = std::nullopt)
        : ctx(std::move(ctx))
        , owned(owned)
        , name(std::move(name))
    {}

    // Reference to main context
    nanobind::ref<Context> ctx;

    // If set the underlying object should be freed on destruction.
    // User created objects normally have this set to true,
    // context/swapchain owned objects have this set to false.
    bool owned = true;

    // Debug name, used in __repr__ and set for vkSetDebugUtilsObjectNameEXT
    std::optional<nanobind::str> name;
};

struct Buffer;
struct Image;
struct Sampler;
struct AccelerationStructure;

struct DescriptorSetBinding: nanobind::intrusive_base {
    DescriptorSetBinding(u32 count, VkDescriptorType type, VkDescriptorBindingFlagBits flags, VkShaderStageFlagBits stage_flags, std::vector<nanobind::ref<Sampler>> immutable_samplers)
        : count(count)
        , type(type)
        , flags((VkDescriptorBindingFlags)flags)
        , stage_flags((VkShaderStageFlags)stage_flags)
        , immutable_samplers(std::move(immutable_samplers))
    { }

    u32 count;
    VkDescriptorType type;
    VkDescriptorBindingFlags flags;
    VkShaderStageFlags stage_flags;
    std::vector<nanobind::ref<Sampler>> immutable_samplers;
};

struct DescriptorPoolSize: VkDescriptorPoolSize {
    DescriptorPoolSize(u32 count, VkDescriptorType type)
        : VkDescriptorPoolSize {
            .type = type,
            .descriptorCount = count,
        }
    {}
};

struct DescriptorSetLayout: GfxObject {
    DescriptorSetLayout(nanobind::ref<Context> ctx, std::vector<nanobind::ref<DescriptorSetBinding>> bindings, VkDescriptorSetLayoutCreateFlagBits flags, std::optional<nanobind::str> name);
    ~DescriptorSetLayout();
    void destroy();

    xpg::gfx::DescriptorSetLayout layout;
    std::vector<nanobind::ref<DescriptorSetBinding>> bindings;
};

struct DescriptorSet;

struct DescriptorPool: GfxObject {
    DescriptorPool(nanobind::ref<Context> ctx, const std::vector<DescriptorPoolSize>& sizes, u32 max_sets, VkDescriptorPoolCreateFlagBits flags, std::optional<nanobind::str> name);
    ~DescriptorPool();
    void destroy();

    nanobind::ref<DescriptorSet> allocate_descriptor_set(nanobind::ref<DescriptorSetLayout> layout, u32 variable_size_count, std::optional<nanobind::str> name);
    void free_descriptor_set(nanobind::ref<DescriptorSet> set);
    void reset();

    xpg::gfx::DescriptorPool pool;
};

struct DescriptorSet: GfxObject {
    DescriptorSet(nanobind::ref<Context> ctx, nanobind::ref<DescriptorPool> pool, xpg::gfx::DescriptorSet set, std::optional<nanobind::str> name);

    void write_buffer(const Buffer& buffer, VkDescriptorType type, u32 binding, u32 element, VkDeviceSize offset, VkDeviceSize size);
    void write_image(const Image& image, VkImageLayout layout, VkDescriptorType type, u32 binding, u32 element);
    void write_combined_image_sampler(const Image& image, VkImageLayout layout, const Sampler& sampler, u32 binding, u32 element);
    void write_sampler(const Sampler& sampler, u32 binding, u32 element);
    void write_acceleration_structure(const AccelerationStructure& as, u32 binding, u32 element);

    nanobind::ref<DescriptorPool> pool;
    xpg::gfx::DescriptorSet set;
};

struct ImFont;
struct Font: nanobind::intrusive_base {
    Font(ImFont* font, nanobind::str name): font(font), name(std::move(name)) {}
    ImFont* font;
    nanobind::str name;
};
