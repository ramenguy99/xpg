#pragma once

// static constexpr usize MAX_ERROR_STACK_SIZE = 64;

// struct Context {
//
// };
//
// struct ErrorStack {
//     Error errors[MAX_ERROR_STACK_SIZE];
// };
// thread_local ErrorStack thread_error_stack;

// template<typename T, typename E>
// struct Result {
//     union {
//         T value;
//         E error;
//     };
//     bool is_error;
//
//     struct OkMarker {};
//     struct ErrMarker {};
//
//     explicit Result(T&& t, OkMarker _)
//         : is_error(false)
//         , value(t)
//     {
//     }
//
//     explicit Result(E&& e, ErrMarker _)
//         : is_error(false)
//         , error(e)
//     {
//     }
//
//
//     Result(const Result& other) = delete;
//     Result& operator=(const Result& other) = delete;
//
//     Result& operator=(Result&& other) {
//         this->is_error = other.is_error;
//         if (is_error) {
//             this->value = std::move(other.value);
//         }
//         else {
//             this->error = std::move(other.error);
//         }
//         return *this;
//     }
//
//     Result(Result&& other) {
//         *this = std::move(other);
//     }
// };
//
// template<typename T, typename E>
// static Result<T, E> Ok(T&& value) {
//     return Result<T, E>(std::move(value), Result<T, E>::OkMarker());
// }
//
// template<typename T, typename E>
// static Result<T, E> Err(E&& error) {
//     return Result<T, E>(std::move(error), Result<T, E>::ErrMarker());
// }
//
// struct Empty {
// };
//
// Result<int, Empty> example_a() {
//     int value = 42;
//
//     auto ok =  Ok<int, Empty>(std::move(value));
//     auto err = Err<int, Empty>(Empty());
//
//     auto hard = Err<int, int>(5);
//
//     return err;
// }
//
// Result<int, Empty> b() {
//     return example_a().value;
// }

