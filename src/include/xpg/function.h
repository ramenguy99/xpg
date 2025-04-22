#pragma once

#include "defines.h"

// Adapted from https://github.com/Pagghiu/SaneCppLibraries
//
// The MIT License (MIT)
//
// Copyright (c) 2022 - present Stefano Cristiano <pagghiu@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copy of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copy or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

struct PlacementNew {};

#if _MSC_VER
inline void* operator new(size_t, void* p, PlacementNew) noexcept { return p; }
inline void  operator delete(void*, void*, PlacementNew) noexcept {}
#else
inline void* operator new(size_t, void* p, PlacementNew) noexcept { return p; }
#endif

/// Placement New
template<typename T, typename... Q> void placementNew(T& storage, Q&&... other) { new (&storage, PlacementNew()) T(forward<Q>(other)...); }

namespace xpg {

template <typename FuncType>
struct Function;

template <typename R, typename... Args>
struct Function<R(Args...)>
{
  private:
    enum class Operation
    {
        Destruct,
        CopyConstruct,
        MoveConstruct
    };
    using ExecuteFunction   = R (*)(const void* const*, typename AddPointer<Args>::type...);
    using OperationFunction = void (*)(Operation operation, const void** other, const void* const*);

    struct VTable
    {
        ExecuteFunction   execute;
        OperationFunction operation;
    };

    static const size_t LAMBDA_PTR_COUNT = 7;
    static const size_t LAMBDA_SIZE = sizeof(void*) * LAMBDA_PTR_COUNT;
    static const size_t FUNCTION_SIZE = sizeof(void*) * (LAMBDA_PTR_COUNT + 1);
    const VTable* vtable;

    union
    {
        const void* classInstance;
        char        lambdaMemory[LAMBDA_SIZE] = {0};
    };

    void checkedOperation(Operation operation, const void** other) const
    {
        if (vtable)
            vtable->operation(operation, other, &classInstance);
    }

  public:
    /// @brief Constructs an empty Function
    Function()
    {
        static_assert(sizeof(Function) == FUNCTION_SIZE, "Function Size");
        vtable = nullptr;
    }

    /// @brief Constructs an empty Function
    Function(decltype(nullptr) ptr)
    {
        vtable = nullptr;
    }

    /// Constructs a function from a lambda with a compatible size (equal or less than LAMBDA_SIZE)
    /// If lambda is bigger than `LAMBDA_SIZE` a static assertion will be issued
    /// SFINAE is used to avoid universal reference from "eating" also copy constructor
    template <
        typename Lambda,
        typename = typename EnableIf<
            not IsSame<typename RemoveReference<Lambda>::type, Function>::value, void>::type>
    Function(Lambda&& lambda)
    {
        vtable = nullptr;
        bind(::forward<typename RemoveReference<Lambda>::type>(lambda));
    }

    /// @brief Destroys the function wrapper
    ~Function() { checkedOperation(Operation::Destruct, nullptr); }

    /// @brief Move constructor for Function wrapper
    /// @param other The moved from function
    Function(Function&& other)
    {
        vtable        = other.vtable;
        classInstance = other.classInstance;
        other.checkedOperation(Operation::MoveConstruct, &classInstance);
        other.checkedOperation(Operation::Destruct, nullptr);
        other.vtable = nullptr;
    }

    /// @brief Copy constructor for Function wrapper
    /// @param other The function to be copied
    Function(const Function& other)
    {
        vtable = other.vtable;
        other.checkedOperation(Operation::CopyConstruct, &classInstance);
    }

    /// @brief Copy assign a function to current function wrapper. Destroys existing wrapper.
    /// @param other The function to be assigned to current function
    Function& operator=(const Function& other)
    {
        checkedOperation(Operation::Destruct, nullptr);
        vtable = other.vtable;
        other.checkedOperation(Operation::CopyConstruct, &classInstance);
        return *this;
    }

    /// @brief Move assign a function to current function wrapper. Destroys existing wrapper.
    /// @param other The function to be move-assigned to current function
    Function& operator=(Function&& other) noexcept
    {
        checkedOperation(Operation::Destruct, nullptr);
        vtable = other.vtable;
        other.checkedOperation(Operation::MoveConstruct, &classInstance);
        other.checkedOperation(Operation::Destruct, nullptr);
        other.vtable = nullptr;
        return *this;
    }

    /// @brief Check if current wrapper is bound to a function
    /// @return `true` if current wrapper is bound to a function
    [[nodiscard]] bool isValid() const { return vtable != nullptr; }

    /// @brief Returns true if this function was bound to a member function of a specific class instance
    [[nodiscard]] bool isBoundToClassInstance(void* instance) const { return classInstance == instance; }

    bool operator==(const Function& other) const
    {
        return vtable == other.vtable and classInstance == other.classInstance;
    }

    /// @brief Binds a Lambda to current function wrapper
    /// @tparam Lambda type of Lambda to be wrapped in current function wrapper
    /// @param lambda Instance of Lambda to be wrapped
    template <typename Lambda>
    void bind(Lambda&& lambda)
    {
        checkedOperation(Operation::Destruct, nullptr);
        vtable = nullptr;
        new (&classInstance, PlacementNew()) Lambda(::forward<Lambda>(lambda));
        vtable = getVTableForLambda<Lambda>();
    }

  private:
    template <typename Lambda>
    static auto getVTableForLambda()
    {
        static_assert(sizeof(Lambda) <= sizeof(lambdaMemory), "Lambda is too big");
        static const VTable staticVTable = {
            [](const void* const* p, typename AddPointer<Args>::type... args)
            {
                Lambda& lambda = *reinterpret_cast<Lambda*>(const_cast<void**>(p));
                return lambda(*args...);
            },
            [](Operation operation, const void** other, const void* const* p)
            {
                Lambda& lambda = *reinterpret_cast<Lambda*>(const_cast<void**>(p));
                if (operation == Operation::Destruct)
                    lambda.~Lambda();
                else if (operation == Operation::CopyConstruct)
                    new (other, PlacementNew()) Lambda(lambda);
                else if (operation == Operation::MoveConstruct)
                    new (other, PlacementNew()) Lambda(move(lambda));
            }};
        return &staticVTable;
    }


  public:
    /// @brief Unsafely retrieve the functor bound previously bound to this function
    /// @tparam Lambda type of Lambda passed to  Function::bind or Function::operator=
    /// @return Pointer to functor or null if Lambda is not the same type bound in bind()
    /// \snippet Libraries/Foundation/Tests/FunctionTest.cpp FunctionFunctorSnippet
    template <typename Lambda>
    Lambda* dynamicCastTo() const
    {
        if (getVTableForLambda<Lambda>() != vtable)
            return nullptr;
        else
            return &const_cast<Lambda&>(reinterpret_cast<const Lambda&>(classInstance));
    }

    /// @brief Binds a free function to function wrapper
    /// @tparam FreeFunction a regular static function to be wrapper, with a matching signature
    template <R (*FreeFunction)(Args...)>
    void bind()
    {
        checkedOperation(Operation::Destruct, nullptr);
        static constexpr const VTable staticVTable = {
            [](const void* const*, typename RemoveReference<Args>::type*... args) constexpr
            { return FreeFunction(*args...); },
            [](Operation, const void**, const void* const*) constexpr {}};
        vtable        = &staticVTable;
        classInstance = nullptr;
    }

    /// @brief Binds a class member function to function wrapper
    /// @tparam Class Type of the Class holding MemberFunction
    /// @tparam MemberFunction Pointer to member function with a matching signature
    /// @param c Reference to the instance of class where the method must be bound to
    template <typename Class, R (Class::*MemberFunction)(Args...) const>
    void bind(const Class& c)
    {
        checkedOperation(Operation::Destruct, nullptr);
        static constexpr const VTable staticVTable = {
            [](const void* const* p, typename RemoveReference<Args>::type*... args) constexpr
            {
                const Class* cls = static_cast<const Class*>(*p);
                return (cls->*MemberFunction)(*args...);
            },
            [](Operation operation, const void** other, const void* const* p) constexpr
            {
                if (operation != Operation::Destruct)
                    *other = *p;
            }};
        vtable        = &staticVTable;
        classInstance = &c;
    }

    /// @brief Binds a class member function to function wrapper
    /// @tparam Class Type of the Class holding MemberFunction
    /// @tparam MemberFunction Pointer to member function with a matching signature
    /// @param c Reference to the instance of class where the method must be bound to
    template <typename Class, R (Class::*MemberFunction)(Args...)>
    void bind(Class& c)
    {
        checkedOperation(Operation::Destruct, nullptr);
        static constexpr const VTable staticVTable = {
            [](const void* const* p, typename RemoveReference<Args>::type*... args) constexpr
            {
                Class* cls = const_cast<Class*>(static_cast<const Class*>(*p));
                return (cls->*MemberFunction)(*args...);
            },
            [](Operation operation, const void** other, const void* const* p) constexpr
            {
                if (operation != Operation::Destruct)
                    *other = *p;
            }};
        vtable        = &staticVTable;
        classInstance = &c;
    }

    /// @brief Invokes the wrapped function. If no function is bound, this is UB.
    /// @param args Arguments to be passed to the wrapped function
    [[nodiscard]] R operator()(Args... args) const { return vtable->execute(&classInstance, &args...); }

    operator bool() const {
        return vtable != nullptr;
    }
};

} // namespace xpg