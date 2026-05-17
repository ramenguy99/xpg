#define TRACER_IMPLEMENTATION
#include "tracer.h"

extern "C" {
#include "testing.h"
}

static size_t page_align(size_t s) {
    size_t ps = ring_buffer_page_size();
    return (s + ps - 1) & ~(ps - 1);
}

// ============================================================
// Ring mapped buffer tests
// ============================================================

TEST(alloc_free) {
    size_t size = page_align(4096);
    void* buf = alloc_ring_mapped_buffer(size);
    CHECK(buf != NULL);
    free_ring_mapped_buffer(buf, size);
}

TEST(write_wraps_to_start) {
    size_t size = page_align(4096);
    uint8_t* buf = (uint8_t*)alloc_ring_mapped_buffer(size);
    CHECK(buf != NULL);

    // Write a pattern that spans the boundary: start at size-32, write 64 bytes
    // The ring mapping means buf[size + x] aliases buf[x]
    for (size_t i = 0; i < 64; i++) {
        buf[size - 32 + i] = (uint8_t)(0xB0 + i);
    }

    // The first 32 bytes went to buf[size-32..size-1] (end of primary)
    // The next 32 bytes went to buf[size..size+31] which aliases buf[0..31]
    for (size_t i = 0; i < 32; i++) {
        CHECK_EQ(buf[i], buf[size + i]);
    }

    // Verify the contiguous read across boundary works
    for (size_t i = 0; i < 64; i++) {
        CHECK_EQ(buf[size - 32 + i], (uint8_t)(0xB0 + i));
    }

    free_ring_mapped_buffer(buf, size);
}

TEST(mirror_consistency) {
    size_t size = page_align(8192);
    uint8_t* buf = (uint8_t*)alloc_ring_mapped_buffer(size);
    CHECK(buf != NULL);

    // Write to primary region
    for (size_t i = 0; i < size; i++) {
        buf[i] = (uint8_t)(i & 0xFF);
    }

    // Mirror region (buf[size..2*size-1]) must reflect the primary
    for (size_t i = 0; i < size; i++) {
        CHECK_EQ(buf[size + i], (uint8_t)(i & 0xFF));
    }

    // Write through the mirror, verify primary reflects it
    buf[size + 100] = 0xDE;
    CHECK_EQ(buf[100], 0xDE);

    free_ring_mapped_buffer(buf, size);
}

TEST(various_sizes) {
    size_t ps = ring_buffer_page_size();
    size_t sizes[] = {ps, ps * 2, ps * 4, ps * 16};
    int n = (int)(sizeof(sizes) / sizeof(sizes[0]));

    for (int i = 0; i < n; i++) {
        void* buf = alloc_ring_mapped_buffer(sizes[i]);
        CHECK(buf != NULL);

        // Write to primary, verify mirror reflects it
        uint8_t* b = (uint8_t*)buf;
        b[sizes[i] - 1] = 0xAA;
        CHECK_EQ(b[2 * sizes[i] - 1], 0xAA);

        // Write through mirror, verify primary reflects it
        b[sizes[i]] = 0xBB;
        CHECK_EQ(b[0], 0xBB);

        free_ring_mapped_buffer(buf, sizes[i]);
    }
}

TEST(contiguous_read_across_boundary) {
    size_t size = page_align(4096);
    uint8_t* buf = (uint8_t*)alloc_ring_mapped_buffer(size);
    CHECK(buf != NULL);

    // Simulate a ring buffer read that starts near the end
    size_t offset = size - 16;
    size_t read_len = 64;

    // Write 64 bytes starting at offset (crosses boundary)
    for (size_t i = 0; i < read_len; i++) {
        buf[offset + i] = (uint8_t)(i + 1);
    }

    // Read contiguously from offset — should work due to mirror
    for (size_t i = 0; i < read_len; i++) {
        CHECK_EQ(buf[offset + i], (uint8_t)(i + 1));
    }

    free_ring_mapped_buffer(buf, size);
}

// ============================================================
// main
// ============================================================

int main() {
    test_setup_timeout(15);

    printf("=== Ring Mapped Buffer Tests ===\n\n");

    RUN(alloc_free);
    RUN(write_wraps_to_start);
    RUN(mirror_consistency);
    RUN(various_sizes);
    RUN(contiguous_read_across_boundary);

    return test_print_results("Ring Buffer Tests");
}
