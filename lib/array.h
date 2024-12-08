#pragma once

#ifndef BOUNDS_CHECKING_ENABLED
#define BOUNDS_CHECKING_ENABLED 1
#endif


template<typename T> struct Array;
template<typename T, usize N> struct ArrayFixed;

template<typename T>
struct ArrayView {
    T* data = 0;
    usize length = 0;

    ArrayView() {}

    ArrayView(T* data, usize length) {
        this->data = data;
        this->length = length;
    }


    ArrayView(const Array<T>& array) {
        data = array.data;
        length = array.length;
    }

    ArrayView& operator=(const Array<T>& array) {
        data = array.data;
        length = array.length;
        return *this;
    }

    template<usize N>
    ArrayView(const ArrayFixed<T, N>& array) {
        data = (T*)array.data;
        length = array.length;
    }

    template<usize N>
    ArrayView& operator=(const ArrayFixed<T, N>& array) {
        data = (T*)array.data;
        length = array.length;
        return *this;
    }


    T& operator[](usize index) {
#if BOUNDS_CHECKING_ENABLED
        if(index >= length) {
            OutOfBounds(index);
        }
#endif
        return data[index];
    }

    void copy_exact(const ArrayView<T>& other) {
        if (length != other.length) {
            OutOfSpace();
        }

        memcpy(data, other.data, length * sizeof(T));
    }

    ArrayView<u8> as_bytes() {
        return ArrayView<u8>((u8*)data, size_in_bytes());
    }

    ArrayView slice(usize index, usize count) {
        if(index + count > length) {
            OutOfBounds(index + count);
        }
        return ArrayView(data + index, count);
    }

    T* consume_ptr(usize count) {
        if(count > length) {
            OutOfBounds(count);
        }
        T* result = data;

        data += count;
        length -= count;

        return result;
    }

    template<typename AdvType>
    AdvType* consume_ptr(usize count = 1)
    {
        //Only allow this on byte like types
        static_assert(sizeof(T) == 1);
        return (AdvType*)consume_ptr(count * sizeof(AdvType));
    }

	template<typename AdvType>
	ArrayView<AdvType> consume_view(usize count = 1)
	{
		//Only allow this on byte like types
		static_assert(sizeof(T) == 1);
		return ArrayView<AdvType>((AdvType*)consume_ptr(count * sizeof(AdvType)), count);
	}

	template<typename AdvType>
	AdvType consume()
	{
		//Only allow this on byte like types
		static_assert(sizeof(T) == 1);
		return *(AdvType*)consume_ptr(sizeof(AdvType));
	}

    template<typename Type>
    Type* as_type() {
        if (sizeof(Type) != length) {
            OutOfBounds(sizeof(Type));
        }
        return (Type*)data;
    }


    bool is_empty() {
        return length == 0;
    }

    usize size_in_bytes() {
        return length * sizeof(T);
    }
};

template<typename T>
struct Array {
    T* data = 0;
    usize length = 0;
    usize capacity = 0;

    Array() {}
    explicit Array(usize initial_length) {
        resize(initial_length);
    }

    Array(const Array& other) = delete;
    Array& operator=(const Array& other) = delete;

    Array& operator=(Array&& other) {
        this->data = other.data;
        this->length = other.length;
        this->capacity = other.capacity;
        other.data = 0;
        other.length = 0;
        other.capacity = 0;
        return *this;
    }

    Array(Array&& other) {
        *this = std::move(other);
    }

    void add(const T& value) {
        if(length == capacity) {
            usize new_capacity = Max<usize>(8, capacity * 2);
            grow_unchecked(new_capacity);
        }

        data[length++] = value;
    }

	void add(T&& value) {
		if(length == capacity) {
			usize new_capacity = Max<usize>(8, capacity * 2);
			grow_unchecked(new_capacity);
		}

		data[length++] = std::move(value);
	}


	void extend(const ArrayView<T>& arr) {
		if(length + arr.length > capacity) {
			usize new_capacity = Max<usize>(8, MAX(capacity * 2, length + arr.length));
			grow_unchecked(new_capacity);
		}

        for (usize i = 0; i < arr.length; i++) {
            data[i + length] = arr.data[i];
        }
		length += arr.length;
	}

    void resize(usize new_size) {
        grow(new_size);
        length = new_size;
    }

    void clear() {
        for (usize i = 0; i < length; i++) {
            data->~T();
        }
        Free(data);
        data = 0;
        capacity = 0;
        length = 0;
    }

    void grow_unchecked(usize new_capacity) {
        T* new_data = (T*)ZeroAlloc(new_capacity * sizeof(T));

        if(length > 0) {
            for (usize i = 0; i < length; i++) {
                new_data[i] = std::move(data[i]);
            }
            Free(data);
        }

        data = new_data;
        capacity = new_capacity;
    }

    void grow(usize new_capacity) {
        if(new_capacity < capacity)
            return;

        grow_unchecked(new_capacity);
    }

    bool is_empty() {
        return length == 0;
    }

    usize size_in_bytes() {
        return length * sizeof(T);
    }

    ArrayView<u8> as_bytes() {
        return ArrayView<u8>((u8*)data, size_in_bytes());
    }

    T& operator[](usize index) {
#if BOUNDS_CHECKING_ENABLED
        if(index >= length) {
            OutOfBounds(index);
        }
#endif
        return data[index];
    }

    const T& operator[](usize index) const {
#if BOUNDS_CHECKING_ENABLED
        if(index >= length) {
            OutOfBounds(index);
        }
#endif
        return data[index];
    }

    ~Array() {
        clear();
    }
};

template<typename T, usize N>
struct ArrayFixed {
    T data[N] = {};
    usize length = 0;

    ArrayFixed() {
    }

    ArrayFixed(usize starting_length) {
        if(length > N) {
            OutOfSpace();
        }
        length = starting_length;
    }

    void add(const T& value) {
        if(length >= N) {
            OutOfSpace();
        }

        data[length++] = value;
    }

    void resize(usize new_size) {
        if(new_size > N) {
            OutOfSpace();
        }

        length = new_size;
    }

    void clear() {
        length = 0;
    }

    T& operator[](usize index) {
#if BOUNDS_CHECKING_ENABLED
        if(index >= length) {
            OutOfBounds(index);
        }
#endif
        return data[index];
    }

    bool is_empty() {
        return length == 0;
    }

    ArrayView<u8> as_bytes() {
        return ArrayView<u8>((u8*)data, size_in_bytes());
    }

    usize size_in_bytes() {
        return length * sizeof(T);
    }
};

template<typename T>
ArrayView<u8> BytesOf(T* obj) {
    return ArrayView<u8>((u8*)obj, sizeof(T));
}

// Similar to an ArrayView, but specifically intended to be passed as a const rvalue reference
// with an initializer list.
template<typename T>
struct Span {
    const T* data;
    size_t length;

    Span(std::initializer_list<T> l) {
        data = l.begin();
        length = l.size();
    }

    Span(const Span& other) = delete;
    Span& operator=(const Span& other) = delete;

    const T& operator[](usize index) const {
#if BOUNDS_CHECKING_ENABLED
        if(index >= length) {
            OutOfBounds(index);
        }
#endif
        return data[index];
    }
};


/*
void printArray(ArrayView<int> a) {
    for(usize i = 0; i < a.length; i++) {
        printf("%d\n", a[i]);
    }
}

void TestArray() {
    ArrayFixed<int, 3> a;
    a.add(5);
    a.add(3);

    ArrayView<int> view = a;

    printArray(a);
    printArray(view);
}
*/
