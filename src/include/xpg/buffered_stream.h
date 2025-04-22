#pragma once

#include "threading.h"
#include "function.h"

namespace xpg {

template<typename T>
struct BufferedStream {
    typedef Function<T(u64, u64, bool)> InitProc;
    typedef Function<bool(T*)> FillProc;

    enum class EntryState: u32 {
        Empty,
        Filling,
        Canceling,
        Done,
    };

    struct Entry {
        alignas(64) std::atomic<EntryState> state;
        alignas(64) T value;
        FillProc fill_proc;

        Entry() {}

        // Obj array requires a move assignment operator, but atomic does not have one.
        Entry(Entry&& other) {
            *this = move(other);
        }

        Entry& operator=(Entry&& other) {
            this->state.store(state.load(std::memory_order_relaxed), std::memory_order_relaxed);
            this->value = move(value);
            this->fill_proc = move(fill_proc);
            return *this;
        }
    };

    ObjArray<Entry> buffer;
    u64 buffer_offset = 0;
    u64 stream_cursor = 0;
    u64 stream_length = 0;

    WorkerPool* pool = nullptr;

    InitProc init_proc;
    FillProc fill_proc;

    static void work_callback(WorkerPool::WorkerInfo* worker_info, void* data) {
        Entry* entry = (Entry*)data;
        while (true) {
            // Check if work has been canceled
            if (entry->state.load(std::memory_order_relaxed) == EntryState::Canceling) {
                entry->state.store(EntryState::Done, std::memory_order_release);
                break;
            }

            // Do a chunk of work
            bool done = entry->fill_proc(&entry->value);

            // When done with work signal ready
            if (done) {
                entry->state.store(EntryState::Done, std::memory_order_release);
                break;
            }
        }
    }

    BufferedStream() {}

    BufferedStream(u64 length, u64 buffer_size, WorkerPool* pool, InitProc init, FillProc fill):
        buffer(Min(length, buffer_size)),
        stream_length(length),
        pool(pool),
        init_proc(init),
        fill_proc(fill) {

        for (u64 i = 0; i < buffer.length; i++) {
            enqueue_load(i, i, false);
        }
    }

    void enqueue_load(u64 buffer_index, u64 stream_index, bool high_priority) {
        Entry* entry = &buffer[buffer_index];

        // Acquire entry, making sure nobody is using it.
        EntryState state = entry->state.load(std::memory_order_relaxed);
        if (state != EntryState::Empty) {
            if (state == EntryState::Filling) {
                // Attempt to cancel pending work.
                // - If this succeeds the worker will exit at some point and we can start waiting.
                // - If this failed the worker just finished, so we can continue.
                entry->state.compare_exchange_strong(state, EntryState::Canceling, std::memory_order_relaxed);
            }

            // Wait for worker to be done with this item.
            while (state != EntryState::Done) {
                state = entry->state.load(std::memory_order_relaxed);
                SpinlockHint();
            }
        }

        // After worker is done, ensure all writes from the worker are complete.
        std::atomic_thread_fence(std::memory_order_acquire);

        // Initialize entry and submit work to pool.
        entry->state.store(EntryState::Filling, std::memory_order_relaxed);
        entry->value = init_proc(stream_index, buffer_index, high_priority);
        entry->fill_proc = fill_proc;

        WorkerPool::Work w;
        w.callback = work_callback;
        w.user_data = entry;
        pool->add_work(w);
    }

    void cancel_all_pending_work() {
        for (u64 i = 0; i < buffer.length; i++) {
            Entry* entry = &buffer[i];

            // Acquire entry, making sure nobody is using it.
            EntryState state = entry->state.load(std::memory_order_relaxed);

            if (state == EntryState::Filling) {
                // Attempt to cancel pending work.
                entry->state.compare_exchange_strong(state, EntryState::Canceling, std::memory_order_relaxed);
            }
        }
    }

    // TODO: this waits if the frame is not ready, we should have a way to
    // first ask for the current frame to be loaded for all streams and then wait for them to complete.
    //
    // Additionally we could have some helpers method to ensure that if we have more
    // than one stream (A, B) the future frames are scheduled as A_0, B_0, A_1, B_1 instead of A_0, A_1, B_0, B_1
    // to help with fifo ordering.
    T get_frame(u64 frame) {
        if (frame >= stream_length) {
            return {};
        }

        // Normalize frame to range stream_cursor-> stream_cursor + stream_length
        u64 normalized_frame = frame;
        if (frame < stream_cursor) {
            normalized_frame = frame + stream_length;
        }

        // If the frame is not in range we need to schedule a load, we schedule only one to minimize latency.
        bool reset = false;
        if (buffer.length < stream_length) {
            if (normalized_frame >= stream_cursor + buffer.length) {
                cancel_all_pending_work();

                reset = true;
                normalized_frame = frame;

                stream_cursor = frame;
                buffer_offset = 0;

                // Schedule current frame
                enqueue_load(0, frame, true);
            }
        }

        // Delta is in range [0, buffer.length[
        u64 delta = normalized_frame - stream_cursor;
        u64 buffer_index = (buffer_offset + delta) % buffer.length;

        // Wait for frame to be ready
        while (buffer[buffer_index].state.load(std::memory_order_relaxed) != EntryState::Done);
        std::atomic_thread_fence(std::memory_order_acquire);

        T value = buffer[buffer_index].value;

        // Schedule more work if needed and update buffer range.
        if (reset) {
            // The buffer was invalid, refresh everything.
            for (u64 i = 1; i < buffer.length; i++) {
                enqueue_load(i, frame + i, false);
            }
        }
        else if (buffer.length < stream_length && delta > 0) {
            // The buffer was valid, refresh missing entries.
            for (u64 i = 0; i < delta; i++) {
                u64 buffer_index = (buffer_offset + buffer.length + i) % buffer.length;
                u64 frame_index = (frame + buffer.length + i) % stream_length;
                enqueue_load(buffer_index, frame_index, false);
            }

            stream_cursor = frame;
            buffer_offset = buffer_index;
        }

        return value;
    }
};

} // namespace xpg