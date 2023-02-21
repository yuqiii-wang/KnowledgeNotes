# Coroutine

"Co" stands for cooperation. A *coroutine* is asked to (or better expected to) willingly suspend its execution to give other co-routines a chance to execute too. 
So a coroutine is about sharing CPU resources (willingly) so others can use the same resource as oneself is using.

In low-level implementation, coroutine mechanism uses additional CPU registers and heap memory to store temporary execution env and local data, rather than freeing them up every time it returns from a callee function.
As a result, the execution env switch between caller and callee can be fast.

In some contexts, coroutines may refer to stackful functions while generators may refer to stackless functions.

### Typical *Call* and *Return* 

A normal function can be thought of as having two operations: *Call* and *Return*.

A call is invocation of another function by a caller function, action including creating an activation frame (the stack of the current running function), suspending execution of the caller function.

The return operation passes the return-value to the caller, destroys the activation frame, and resume caller function execution.

* Call

In a call of another function, the caller should suspend itself (typically, saving data currently held in CPU registers so that those values can later be restored).

* Return

When a function returns, the caller function destroys any local variables in-scope at the return-point, and free memory used by the activation-frame.

The caller function restores its activation frame by setting the stack register to point to the activation frame of its own.
Then resume execution from where it was suspended.

### Coroutines' Suspend, Resume and Destroy

Coroutines replace call and return operations with *Suspend*, *Resume* and *Destroy*.

* Suspend

There is no activation frame destroy when transferring execution between caller and callee.

In the C++ Coroutines, these suspend-points are identified by usages of the `co_await` or `co_yield` keywords.

* Resumption

The Resume operation can be performed on a coroutine that is currently in the ‘suspended’ state.

Instead of transferring execution to the start of the function, coroutine will transfer execution to the point in the function at which it was last suspended. 
It does this by loading the resume-point from the coroutine-frame and jumping to that point.

* Destroy

The Destroy operation destroys the coroutine frame without resuming execution of the coroutine.

Similar to "return" of a normal function, destroy removes activation frame and frees local/out-of-scope memory.

## Coroutine General Execution

### Each coroutine is associated with

* the *promise object*, manipulated from inside the coroutine. The coroutine submits its result or exception through this object.
* the *coroutine handle*, manipulated from outside the coroutine. This is a non-owning handle used to resume execution of the coroutine or to destroy the coroutine frame.
* the *coroutine state/frame*, which is an internal, heap-allocated (unless the allocation is optimized out), object that contains
  * the promise object
  * the parameters (all copied by value)
  * some representation of the current suspension point, so that a resume knows where to continue, and a destroy knows what local variables were in scope
  * local variables and temporaries whose lifetime spans the current suspension point.
  When a coroutine begins execution, it performs the following:

* allocates the coroutine state/frame object using `operator new`.
* copies all function parameters to the coroutine state/frame: by-value parameters are moved or copied, by-reference parameters remain references (thus, may become dangling, if the coroutine is resumed after the lifetime of referred object ends).

* calls the constructor for the promise object. If the promise type has a constructor that takes all coroutine parameters, that constructor is called, with post-copy coroutine arguments. Otherwise the default constructor is called.
* calls `promise.get_return_object()` and keeps the result in a local variable. The result of that call will be returned to the caller when the coroutine first suspends.
* calls `promise.initial_suspend()` and `co_awaits` its result. Typical Promise types either return a `std::suspend_always`, for lazily-started coroutines, or `std::suspend_never`, for eagerly-started coroutines.
* when `co_await promise.initial_suspend()` resumes, starts executing the body of the coroutine.

### When a coroutine reaches a suspension point

The return object obtained earlier is returned to the caller/resumer, after implicit conversion to the return type of the coroutine, if necessary.

### When a coroutine reaches the `co_return` statement, it performs the following:

* calls `promise.return_void()`
* destroys all variables with automatic storage duration in reverse order they were created.
* calls `promise.final_suspend()` and `co_awaits` the result.

### Exception

* call `promise.unhandled_exception()` from within the catch-block
* calls `promise.final_suspend()` and `co_awaits` the result


## Coroutine Interface - Awaiter

A coroutine awaiter is the flow controller to determine if a coroutine should suspend/resume.

An awaiter has three member functions: `await_ready`, `await_suspend` and `await_resume`.

### Awaiter Structure

Suspension is done through `co_await` that works on `struct awaiter`.

Typically, `struct awaiter` has three member functions:

* `bool await_ready();` checks if awaiter is ready to be executed without suspension

`std` provides the below implementation.
```cpp
struct suspend_never {
    constexpr bool await_ready() const noexcept {
        return true;  //no suspension
    }
    ...
};

struct suspend_always {
    constexpr bool await_ready() const noexcept {
        return false; // should suspend
    }

    ...
};
```

* `await_suspend(std::coroutine_handle<> coroutine_handle);`

When `await_ready()` returns false, should suspend.

Suspension means storing local environments into a list/vector that is stored in heap without destroying it, so that previous local environments can be recovered fast.

`coroutine_handle` acts as a bridge to connect awaiter and promise.

`coroutine_handle.resume();` is used to recover the coroutine execution.

* `await_resume();` starts the previously suspended coroutine execution 

Typically, used `co_await` as the return value.

### An Example Awaiter

```cpp
struct Awaiter {
  int value;

  bool await_ready() { // coroutine suspension
    return false;
  }

  void await_suspend(std::coroutine_handle<> coroutine_handle) {
    // switch to another thread
    std::async([=](){
      using namespace std::chrono_literals;
      // sleep 1s
      std::this_thread::sleep_for(1s); 
      // coroutine resumes
      coroutine_handle.resume();
    });
  }

  int await_resume() {
    // value is the co_await return value
    return value;
  }
};
```

Invoked by
```cpp
Result Coroutine() {
  std::cout << 1 << std::endl;
  std::cout << co_await Awaiter{.value = 1000} << std::endl;
  std::cout << 2 << std::endl; // runs after 1 sec
};
```

## Coroutine Interface - Promise

To distinguish a coroutine, should judge its return value. 
If satisfied the coroutine rule, the function will be compiled into a coroutine.

The rule is `coroutine_traits`.
```cpp
template <class _Ret, class = void>
struct _Coroutine_traits {};

template <class _Ret>
struct _Coroutine_traits<_Ret, void_t<typename _Ret::promise_type>> {
    using promise_type = typename _Ret::promise_type;
};

template <class _Ret, class...>
struct coroutine_traits : _Coroutine_traits<_Ret> {};
```

In other words, if a return type `_Ret` can find a `_Ret::promise_type`, the function that has this `_Ret` can be said a coroutine.

Denote `Result` as a return type, it should have this definition
```cpp
struct Result {
  struct promise_type {
    ...
  };
};
```

A promise controls the actual coroutine execution detail, such as what the first time suspension should do (defined in `initial_suspend`) and clean-up work defined in `final_suspend`.

### Promise Structure 

Given this coroutine 
```cpp
Result Coroutine() {
  std::cout << 1 << std::endl;
  std::cout << co_await Awaiter{.value = 1000} << std::endl;
  std::cout << 2 << std::endl; // runs after 1 sec
};
```
coroutine return value `struct Result` is defined with a contained `struct promise_type`, where `get_return_object()` is used to handle `Result`.

```cpp
struct Result {
  struct promise_type {

    Result get_return_object() {
      return {};
    }

    ...
  };
};
```

Different from other normal function, coroutines' return value is created soon after coroutine frame is created (construct `promise_type` and `get_return_object` to construct return value object).

Once coroutine return value `struct Result` is constructed, coroutine starts execution.

* `initial_suspend` is first invoked `co_await promise.initial_suspend()` that returns an awaiter.
If suspension condition is reached, coroutine is suspended 

* `return_value(T t);` corresponds to `co_return t`;
* `return_void();` corresponds to `co_return`;

* Exception can be handled by

```cpp
void unhandled_exception() {
  exception_ = std::current_exception(); 
}
```

* `final_suspend` is called to clean up the coroutine frame, such as destroying the coroutine frame.


## Symmetric Coroutines and Asymmetric Coroutines

Symmetric coroutine facilities provide a single control-transfer operation that allows coroutines to explicitly pass control between themselves.  

Asymmetric coroutine mechanisms (more commonly denoted as semi-symmetric or semi coroutines) provide two control-transfer operations: one for invoking a coroutine and one for suspending it, the latter returning control to the coroutine invoker. 

## "Schedule" Concept in Coroutine

Unlike Python, C++ does not have a built-in event loop. To "schedule" a coroutine to resume on a thread, should manually indicate where and when to run `co_await` or `co_yield` .