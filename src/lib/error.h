
// Error on the handle utility.
// 
// Idea here is that you add one of these to your context struct,
// at the start of every function you check if the error is signaled,
// if it is return a no-op.
// 
// Wherever you want to do error handling you first check that the error
// is clear, call the function you want to call and then check the error again.
// If errored you can push extra context about the error that any caller 
// can then use to give meaningful error messages or to better handle the error.
//
// We want to provide macros / utils for the most common operations, but the main
// goal is to minimize / standardize places where you must do error handling.

template <typename T>
struct Error {
    struct Context {
        T value;
        char msg[256];
    };

    bool errored;

    // Stack of error contexts, first error is deepest in call stack.
    ArrayFixed<Context, 16> context;
};