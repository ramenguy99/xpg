#pragma once

#include <string.h>
#include <initializer_list>
#include <type_traits>

#include "defines.h"

#ifndef BOUNDS_CHECKING_ENABLED
#define BOUNDS_CHECKING_ENABLED 1
#endif

namespace xpg {

template<typename T, usize ALIGN=alignof(T)> struct Array;
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


    template<usize ALIGN>
    ArrayView(const Array<T, ALIGN>& array) {
        data = array.data;
        length = array.length;
    }

    template<usize ALIGN>
    ArrayView& operator=(const Array<T, ALIGN>& array) {
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

	template<typename Type>
	ArrayView<Type> as_view()
	{
        if (length % sizeof(Type) != 0) {
            OutOfSpace();
        }
		return ArrayView<Type>((Type*)data, length / sizeof(Type));
	}


    bool is_empty() {
        return length == 0;
    }

    usize size_in_bytes() {
        return length * sizeof(T);
    }
};

template<typename T>
struct ObjArray {
    T* data = 0;
    usize length = 0;
    usize capacity = 0;

    ObjArray() {}
    explicit ObjArray(usize initial_length) {
        resize(initial_length);
    }

    ObjArray(const ObjArray& other) = delete;
    ObjArray& operator=(const ObjArray& other) = delete;

    ObjArray& operator=(ObjArray&& other) {
        AlignedFree(this->data);

        this->data = other.data;
        this->length = other.length;
        this->capacity = other.capacity;
        other.data = 0;
        other.length = 0;
        other.capacity = 0;
        return *this;
    }

    ObjArray(ObjArray&& other) {
        *this = move(other);
    }

    void add(const T& value) {
        if(length == capacity) {
            usize new_capacity = Max<usize>(8, capacity * 2);
            grow_unchecked(new_capacity);
        }

        new(data + length++) T(value);
    }

	void add(T&& value) {
		if(length == capacity) {
			usize new_capacity = Max<usize>(8, capacity * 2);
			grow_unchecked(new_capacity);
		}

        new(data + length++) T(move(value));
	}


	void extend(const ArrayView<T>& arr) {
		if(length + arr.length > capacity) {
			usize new_capacity = Max<usize>(8, MAX(capacity * 2, length + arr.length));
			grow_unchecked(new_capacity);
		}

        // SUPPORT_NON_POD
        for (usize i = 0; i < arr.length; i++) {
            new(data + length++) T(move(arr.data[i]));
        }
        //

		length += arr.length;
	}

    void resize(usize new_size) {
        grow(new_size);
        for (usize i = length; i < new_size; i++) {
            new (data + i) T();
        }
        length = new_size;
    }

    void clear() {
        // SUPPORT NON POD
        for (usize i = 0; i < length; i++) {
            data[i].~T();
        }
        //

        AlignedFree(data);
        data = 0;
        capacity = 0;
        length = 0;
    }

    void grow_unchecked(usize new_capacity) {
        T* new_data = (T*)AlignedAlloc(alignof(T), new_capacity * sizeof(T));

        // SUPPORT NON POD
        for (usize i = 0; i < length; i++) {
            new(new_data + i) T(move(data[i]));

            // Destroy old element (even if we moved out we should still destroy the old object)
            data[i].~T();
        }
        //

        AlignedFree(data);
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

    ArrayView<T> slice(usize index, usize count) {
        return ArrayView(*this).slice(index, count);
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

    ~ObjArray() {
        clear();
    }
};

template<typename T, usize ALIGN>
struct Array {
    static_assert(std::is_trivially_destructible<T>());
    static_assert(std::is_trivially_copyable<T>());

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
        AlignedFree(this->data);

        this->data = other.data;
        this->length = other.length;
        this->capacity = other.capacity;
        other.data = 0;
        other.length = 0;
        other.capacity = 0;
        return *this;
    }

    Array(Array&& other) {
        *this = move(other);
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

		data[length++] = move(value);
	}


	void extend(const ArrayView<T>& arr) {
		if(length + arr.length > capacity) {
			usize new_capacity = Max<usize>(8, MAX(capacity * 2, length + arr.length));
			grow_unchecked(new_capacity);
		}

        memcpy(data + length, arr.data, arr.length * sizeof(T));

		length += arr.length;
	}

    void resize(usize new_size) {
        grow(new_size);
        length = new_size;
    }

    void clear() {
        AlignedFree(data);
        data = 0;
        capacity = 0;
        length = 0;
    }

    void grow_unchecked(usize new_capacity) {
        T* new_data = (T*)AlignedAlloc(ALIGN, new_capacity * sizeof(T));

        memcpy(new_data, data, length * sizeof(T));
        memset(new_data + length, 0, (new_capacity - length) * sizeof(T));
        AlignedFree(data);

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

    ArrayView<T> slice(usize index, usize count) {
        return ArrayView(*this).slice(index, count);
    }

    bool contains(const T &value) {
        for (usize i = 0; i < length; i++) {
            if (data[i] == value) {
                return true;
            }
        }
        return false;
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

    ArrayView<T> slice(usize index, usize count) {
        return ArrayView(*this).slice(index, count);
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

    Span(ArrayView<T> view) {
        data = view.data;
        length = view.length;
    }

    Span(const Array<T>& a) {
        data = a.data;
        length = a.length;
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

} // namespace xpg