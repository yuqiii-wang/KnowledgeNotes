# Functions



## Function Pointer

## Lambda

## `std::function` and `std::bind`

## Examples

* order of execution

```cpp
int (*((*ptr(int, int)))) (int); 
```

Explain:
```cpp
// function return to a pointer
*ptr(int, int)

// take the return pointer as an arg
(*ptr(int, int))

// extra parentheses does not make any difference
((*ptr(int, int)))

// function pointer to pointer
*((*ptr(int, int)))

// function pointer to int pointer
int (*((*ptr(int, int))))
```