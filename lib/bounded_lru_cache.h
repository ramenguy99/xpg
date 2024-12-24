template <typename T>
struct BoundedLRUCache {
    struct Entry {
        Entry* next;
        Entry* prev;
        T value;
    };

    Array<Entry> nodes;
    Entry* next_free;

    Entry* head = 0;
    Entry* tail = 0;

    BoundedLRUCache(usize length): nodes(length) {
        next_free = nodes.data;
        for (usize i = 1; i < nodes.length; i++) {
            nodes[i - 1].next = &nodes[i];
        }
    }

    Entry* add(T&& value) {
        if (next_free == 0) {
            pop();
        }

        // Pop from free list
        Entry* e = next_free;
        next_free = next_free->next;

        // Insert new element at the head
        e->value = std::move(value);
        e->next = 0;
        e->prev = head;

        // Update old head to point to new head, if it exists
        if (head) {
            head->next = e;
        }

        // Set new head
        head = e;

        // If queue was empty, also set tail
        if (!tail) {
            tail = e;
        }

        return e;
    }

    void pop() {
        if (tail) {
            Entry* e = tail;

            // Add entry to free list
            e->next = next_free;
            e->prev = 0;
            next_free = e;

            // Update tail
            tail = e->next;

            // Update prev link on new tail, if it exists
            if(tail) {
                tail->prev = 0;
            }
        }
    }

    void remove(Entry* e) {
        // Load links
        Entry* next = e->next;
        Entry* prev = e->prev;

        // Pop from list
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

        // Push to free list
        e->next = next_free;
        e->prev = 0;
        next_free = e;
    }
};
