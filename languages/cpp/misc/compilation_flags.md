# Some Compilation Flags

### `-fpermissive` and `-Wno-changes-meaning`

Under the same scope, a variable should not see its definition changes.

```cpp
struct A;
struct B1 { A a; typedef A A; }; // warning, 'A' changes meaning
struct B2 { A a; struct A { }; }; // error, 'A' changes meaning
```

This compilation error can be reduce to just a warning if compiled with the flags `-fpermissive` or `-Wno-changes-meaning`.