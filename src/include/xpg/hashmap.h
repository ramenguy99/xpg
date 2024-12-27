#pragma once

#include <defines.h>

inline u64 Hash(u64 x) 
{
    x *= 0xff51afd7ed558ccd;
    x ^= x >> 32;
    return x;
}

inline u64 Hash(u8  x) { return Hash((u64)x); }; 
inline u64 Hash(u16 x) { return Hash((u64)x); };  
inline u64 Hash(u32 x) { return Hash((u64)x); };  
inline u64 Hash(s8  x) { return Hash((u64)x); };  
inline u64 Hash(s16 x) { return Hash((u64)x); };  
inline u64 Hash(s32 x) { return Hash((u64)x); };  
inline u64 Hash(s64 x) { return Hash((u64)x); };  

template<typename T>
u64 Hash(T* p)
{
    return Hash((u64)p);
}

inline u64 HashBytes(u8* bytes, usize length) {
    u64 x = 0xcbf29ce484222325;
    for (usize i = 0; i < length; i++) {
        x ^= bytes[i];
        x *= 0x100000001b3;
        x ^= x >> 32;
    }
    return x;
}

template<typename K, typename V>
struct HashMap {
    static_assert(std::is_trivially_destructible<K>());
    static_assert(std::is_trivially_copyable<K>());
    static_assert(std::is_trivially_destructible<V>());
    static_assert(std::is_trivially_copyable<V>());
    static_assert(std::is_move_assignable<V>());
    static_assert(std::is_move_constructible<V>());

    static constexpr u64 HASH_EMPTY = 0;

    u64* hashes = 0;
    K* keys = 0;
    V* values = 0;
    usize count = 0;
    usize capacity = 0;

    HashMap() {}
    explicit HashMap(usize initial_size) {
        assert(IsPow2(initial_size));
        grow(Max<usize>(8, initial_size));
    }

    ~HashMap() {
        Free(hashes);
        Free(keys);
        Free(values);
    }

    usize hash_key(const K& k) {
        u64 hash = Hash(k);
        if (k == HASH_EMPTY) return HASH_EMPTY + 1;
        else return hash;
    }

    void grow(usize new_capacity) {
        assert(IsPow2(new_capacity));
        if (new_capacity <= capacity) return;

        u64* new_hashes = (u64*)ZeroAlloc(new_capacity * sizeof(u64));
        K* new_keys = (K*)ZeroAlloc(new_capacity * sizeof(K));
        V* new_values = (V*)ZeroAlloc(new_capacity * sizeof(V));

        u64 mask = new_capacity - 1;
        for (usize i = 0; i < capacity; i++) {
            u64 h = hashes[i];
            if (h == HASH_EMPTY) continue;

            usize index = h & mask;
            new_hashes[index] = h;
            new_keys[index] = std::move(keys[i]);
            new_values[index] = std::move(values[i]);
        }

        hashes = new_hashes;
        keys = new_keys;
        values = new_values;
        capacity = new_capacity;
    }

    void insert(const K& k, V&& v) {
        if (count <= capacity >> 1) {
            grow(Max<usize>(8, capacity * 2));
        }

        usize mask = capacity - 1;
        u64 h = hash_key(k);
        usize index = h & mask;
        while (true) {
            u64 prev_hash = hashes[index];
            if (prev_hash == HASH_EMPTY) {
                // Fill empty slot
                hashes[index] = h;
                keys[index] = k;
                values[index] = std::move(v);
                count += 1;
            } else {
                // Check if same key
                if (prev_hash == h && keys[index] == k) {
                    // Replace existing slot
                    values[index] = std::move(v);
                }
                else {
                    // Continue search
                    index = (index + 1) & mask;
                }
            }
        }
    }

    V* get(const K& k) {
        if(capacity == 0) return 0;

        usize mask = capacity - 1;
        u64 h = hash_key(k);
        usize index = h & mask;
        while (true) {
            u64 prev_hash = hashes[index];
            if (prev_hash == HASH_EMPTY) {
                // Key not found
                return 0;
            } else {
                // Check if same key
                if (prev_hash == h && keys[index] == k) {
                    // Return pointer to existing slot
                    return &values[index];
                }
                else {
                    // Continue search
                    index = (index + 1) & mask;
                }
            }
        }
    }
};