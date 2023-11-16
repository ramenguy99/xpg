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
            return 0;
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
	AdvType consume(usize count = 1)
	{
		//Only allow this on byte like types
		static_assert(sizeof(T) == 1);
		return *(AdvType*)consume_ptr(count * sizeof(AdvType));
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
    Array(usize initial_length) {
        resize(initial_length);
    }
    
    Array(const Array& other) = delete;
    Array& operator=(const Array& other) = delete;
    
    Array& operator=(Array&& other) {
        this->data = other.data;
        this->length = other.length;
        other.data = 0;
        other.length = 0;
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

		memcpy(data + length, arr.data, arr.length * sizeof(T));
		length += arr.length;
	}
    
    void resize(usize new_size) {
        grow(new_size);
        length = new_size;
    }
    
    void clear() {
        Free(data);
        capacity = 0;
        length = 0;
    }
    
    void grow_unchecked(usize new_capacity) {
        T* new_data = (T*)ZeroAlloc(new_capacity * sizeof(T));
        
        if(length > 0) {
            memcpy(new_data, data, length * sizeof(T));
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

    ~Array() {
        Free(data);
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
        if(new_size >= N) {
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
