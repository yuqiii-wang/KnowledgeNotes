# Virtual Table and Inheritance

Every virtual-contained class has a per-class virtual table.

```cpp
class BaseNoVirtual
{
public:
  void foo() {};
};

class Base {
 public:
  virtual void foo() {};
};

std::cout << "Size of BaseNoVirtual: " << sizeof(BaseNoVirtual) << std::endl;
std::cout << "Size of Base: " << sizeof(Base) << std::endl;
```

The above code would print
```bash
Size of BaseNoVirtual: 1
Size of Base: 8
```

Every `class` even empty should occupy some mem, hence having 1 byte of data.
`Base` is of size 8 bytes (on 64-bit machine) for it has a hidden pointer pointing to a `vtable`.
The 8 bytes is the size of a pointer.

Run with `gdb`, 