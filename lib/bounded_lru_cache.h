template<typename T>
struct Pool {
    ObjArray<Array<T>> blocks;
    usize block_size;
    
    Pool(usize initial_count, usize block_size = 1024): block_size(block_size) {
        Array<T> a;
        a.grow(Max(initial_count, block_size));
        blocks.add(std::move(a));
    }

    T* alloc() {
        {
            Array<T>& last = blocks[blocks.length - 1];
            if (last.length < last.capacity) {
                Array<T> a;
                a.grow(block_size);
                blocks.add(std::move(a));
            }
        }

        Array<T>& last = blocks[blocks.length - 1];
        assert(last.length < last.capacity);

        last.add({});
        return last.data + last.length - 1;
    }
};

template <typename T>
struct BoundedLRUCache {
    struct Entry {
        Entry* next;
        Entry* prev;
        T value;
    };

    // Backing storage for nodes
    Pool<Entry> nodes;
    Entry* next_free = 0;

    // LRU queue
    Entry* head = 0;
    Entry* tail = 0;

    BoundedLRUCache(usize length): nodes(length) { }

    Entry* alloc(T&& value) {
        if (next_free == 0) {
            next_free = nodes.alloc();
        }

        // Pop from free list
        Entry* e = next_free;
        next_free = next_free->next;

        // Insert new element at the head
        e->value = std::move(value);
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
