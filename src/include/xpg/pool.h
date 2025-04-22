#pragma once

#include "array.h"

namespace xpg {

// A grow-only pool allocator.
//
// This allocator provides two main features:
// 1. Allocations are batched into blocks of configurable size.
// 2. Allocations are guaranteed to be stable, e.g. pointers that are returned will always be valid as long as the allocator lives.
//
// This is a useful building blocks for creating more
// useful freelist-like allocators.
template<typename T>
struct PoolAllocator {
    ObjArray<Array<T>> blocks;
    usize block_size;

    PoolAllocator(usize initial_count, usize block_size = 1024): block_size(block_size) {
        Array<T> a;
        a.grow(Max(initial_count, block_size));
        blocks.add(move(a));
    }

    T* alloc() {
        {
            Array<T>& last = blocks[blocks.length - 1];
            if (last.length < last.capacity) {
                Array<T> a;
                a.grow(block_size);
                blocks.add(move(a));
            }
        }

        Array<T>& last = blocks[blocks.length - 1];
        assert(last.length < last.capacity);

        last.add({});
        return last.data + last.length - 1;
    }
};

// A free list of nodes backed by a pool allocator and a FIFO queue.
//
// These two data structures are combined to allow sharing links
// and allowing to store links inline with the user value.
template <typename T>
struct PoolQueue {
    struct Entry {
        Entry* next;
        Entry* prev;
        T value;
    };

    // Backing storage for nodes
    PoolAllocator<Entry> nodes;
    Entry* next_free = 0;

    // FIFO queue
    Entry* head = 0;
    Entry* tail = 0;

    PoolQueue(usize length): nodes(length) { }

    Entry* alloc(T&& value) {
        if (next_free == 0) {
            next_free = nodes.alloc();
        }

        // Pop from free list
        Entry* e = next_free;
        next_free = next_free->next;

        // Insert new element at the head
        e->value = move(value);
        e->next = 0;
        e->prev = 0;

        return e;
    }

    void free(Entry* e) {
        assert(e);
        assert(e->next == 0);
        assert(e->prev == 0);

        if (e) {
            e->next = next_free;
            next_free = e;
        }
    }

    void push(Entry* e) {
        // If queue is empy also make this the new head
        if (head == 0) {
            head = e;
        }

        // Update link on existing tail, if it exists
        if (tail) {
            tail->prev = e;
        }

        // Link element to new tail
        e->next = tail;

        // Update tail
        tail = e;
    }

    // Pops an element from the head of the queue, this is the element that was inserted least recently.
    Entry* pop() {
        if (head) {
            Entry* e = head;

            // Update head
            head = e->prev;

            // Update next link on new head, if it exists
            if(head) {
                head->next = 0;
            }

            // Reset links, node is now outside of list.
            e->prev = 0;

            // If the queue is empty after popping, also update the tail
            if (tail == e) {
                tail = 0;
            }

            return e;
        }
        else {
            return 0;
        }
    }

    void remove(Entry* e) {
        // Load links
        Entry* next = e->next;
        Entry* prev = e->prev;

        // Remove from queue
        if (next) {
            next->prev = prev;
        }
        else {
            head = prev;
        }

        if (prev) {
            prev->next = next;
        }
        else {
            tail = next;
        }

        // Reset links, node is now outside of list.
        e->next = 0;
        e->prev = 0;
    }
};

} // namespace xpg