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

struct DescriptorSetEntry: xpg::gfx::DescriptorSetEntryDesc {
    DescriptorSetEntry(u32 count, VkDescriptorType type)
        : xpg::gfx::DescriptorSetEntryDesc {
            .count = count,
            .type = type
        }
    {
    };
};
static_assert(sizeof(DescriptorSetEntry) == sizeof(xpg::gfx::DescriptorSetEntryDesc));

struct Buffer;
struct Image;
struct Sampler;
struct AccelerationStructure;

struct DescriptorSet: GfxObject {
    DescriptorSet(nanobind::ref<Context> ctx, const std::vector<DescriptorSetEntry>& entries, VkDescriptorBindingFlagBits flags, std::optional<nanobind::str> name);
    void write_buffer(const Buffer& buffer, VkDescriptorType type, u32 binding, u32 element, VkDeviceSize offset, VkDeviceSize size);
    void write_image(const Image& image, VkImageLayout layout, VkDescriptorType type, u32 binding, u32 element);
    void write_combined_image_sampler(const Image& image, VkImageLayout layout, const Sampler& sampler, u32 binding, u32 element);
    void write_sampler(const Sampler& sampler, u32 binding, u32 element);
    void write_acceleration_structure(const AccelerationStructure& as, u32 binding, u32 element);
    ~DescriptorSet();
    void destroy();
    xpg::gfx::DescriptorSet set;
};

struct ImFont;
struct Font: nanobind::intrusive_base {
    Font(ImFont* font, nanobind::str name): font(font), name(std::move(name)) {}
    ImFont* font;
    nanobind::str name;
};
