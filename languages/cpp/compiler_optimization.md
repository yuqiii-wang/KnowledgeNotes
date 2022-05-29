# Optimization


## Return Value 

Compiler can optimize the "copy" return value by such as *placement new*, and it is discouraged using `std::move` as the lifecycle of local memory is not guaranteed.

This method is referred to as *copy elision*: a compiler optimization technique that eliminates unnecessary copying of objects.

```cpp
Widget makeWidget()
{
Widget w;
return w; // "copy" w into return value
}
```
vs
```cpp
Widget makeWidget()
{
Widget w;
return std::move(w);// Moving version of makeWidget
                    // move w into return value
                    // (don't do this!)
}
```

## Small String Optimization (SSO)

When a string length is smaller than 20, data is stored on stack rather than on heap, as heap storage requires additional operation costs such as `new`.

However, from user side, a `std::string` works consistently regardless of its length.