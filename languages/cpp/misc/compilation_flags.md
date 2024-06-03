# Some Compilation Flags

* `-Wall -Werror`

Show compilation time warning and error.

For example, below code would raise a warning

```cpp
int* foo() {
  int a = 0;
  return &a;
}
```

* `-fpermissive` and `-Wno-changes-meaning`

Under the same scope, a variable should not see its definition changes.

```cpp
struct A;
struct B1 { A a; typedef A A; }; // warning, 'A' changes meaning
struct B2 { A a; struct A { }; }; // error, 'A' changes meaning
```

This compilation error can be reduce to just a warning if compiled with the flags `-fpermissive` or `-Wno-changes-meaning`.