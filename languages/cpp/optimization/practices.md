# Common optimization practices

## Memory access by block 

```cpp
// multiple threads running a lambda
int a[N];
std::thread* threadRunsPtrs[n];
for (int i = 0; i < n; i++)
{
    threadRunsPtrs[i] = new thread(lambdaPerThread, i, N/n);
}

// each thread accessing an individual block of mem,
// good for parallel computation
auto lambdaPerThread = [&](int threadId; int blockRange)
{
    for (int i = blockRange*threadId; i < blockRange*(threadId+1); i++)
    {
        a[i];
    }
}

// scattered mem access, bad for parallel computation
// actually only one thread is running at a time
auto lambdaPerThread = [&](int threadId; int threadTotalNum)
{
    for (int i = threadId; i < N; i += threadTotalNum)
    {
        a[i];
    }
}
```

## Inline

`inline` function are faster in execution( compared to normal function) due to overhead saved by removal of
* function call
* pushing of function parameters on stack

However, it might reduce performance if misused, for increased cache misses and thrashing.

### `inline` Implementation

Inline expansion is similar to macro expansion as the compiler places a new copy of the function in each place it is called. 
Inlined functions run a little faster than the normal functions as function-calling-overheads are saved.

Inline expansion is used to eliminate the time overhead (excess time) when a function is called.

Without inline functions, the compiler decides which functions to inline.

Ordinarily, when a function is invoked, control is transferred to its definition by a branch or call instruction. With inlining, control drops through directly to the code for the function, without a branch or call instruction.

`inline` does not work for virtual function nor recursion.

### Implicit `inline`

When there is need to `sort` many `struct S`, the below code is inefficient for `compare` will not be considered `inline`.
```cpp
struct S {
    int a, b;
};
int n_items = 10000;
S arrS[n_items];

bool compare(const S& s1, const S& s2) {
    return s1.b < s2.b;
}

std::sort(arrS, arrS + n_items, compare);
```

Instead, define `struct Comparator`. The `operator()` is by default `inline`
```cpp
struct Comparator {
    bool operator()(const S& s1, const S& s2) const {
        return s1.b < s2.b;
    }
};

std::sort(arr, arr + n_items, Comparator());
```

### `inline` vs Macro

Inline expansion  occurs during compilation, without changing the source code (the text), while macro expansion occurs prior to compilation.

## `noexcept`

Compiler uses *flow graph* to optimize machine code generation. A flow graph consists of what are generally called "blocks" of the function (areas of code that have a single entrance and a single exit) and edges between the blocks to indicate where flow can jump to. `noexcept` alters the flow graph (simplifies flow graph not to cope with any error handling)

For example, code below using containers might throw `std::bad_alloc` error for lack of memory, adding complexity to flow graph. 
There are many errors a function can throw, and error handling code blocks can be many in a flow graph. By `noexcept`, flow graph is trimmed such that only `std::terminate()` is invoked when error throws. 
```cpp
double compute(double x) noexcept {
    std::string s = "Courtney and Anya";
    std::vector<double> tmp(1000);
    // ...
}
```

Another example is that, containers such as `std::vector` will move their elements if the elements' move constructor is `noexcept`, 
and copy otherwise (unless the copy constructor is not accessible, but a potentially throwing move constructor is, in which case the strong exception guarantee is waived).

### `noexcept` Best Practices

Use `noexcept` in below scenarios:
* move constructor
* move assignment
* destructor (since C++11, they are by default `noexcept`)
* 