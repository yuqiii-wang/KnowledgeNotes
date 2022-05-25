# How an executable works

How an executable is produced:

![GCC_CompilationProcess](imgs/GCC_CompilationProcess.png "GCC_CompilationProcess")

## Linker

Program components are usually NOT contained within a single object file (composed of machine code out of assembler/compiler). Components reference each other by means of symbol (debug symbols, attached additional symbol information to symbol table)

Linker takes care of arranging object addresses for reference by main entry file.

### Dynamic Linker

Dynamic linker loads and links object files to executable at run time, by copying the content of libraries from persistent storage to RAM, filling jump tables and relocating pointers.

### Static Linker

All program object references are resolved at compile time and and copied into an executable, and when the executable runs, objects are referenced simply by address offsetting loading into RAM.

Static executable is discouraged provided that it is built against system archive libraries, once built, future OS updates might cause failure. 

### Example (static lib)

Compile from source code `fn1.cpp` and `fn2.cpp`:
```bash
gcc -c fn1.cpp fn2.cpp
```
which produces `fn1.o` and `fn2.o`.

Then output a static lib `static.a`:
```bash
ar rcs static.a fn1.o fn2.o
```

Check the static lib:
```bash
nm static.a
ar -t static.a
```

Get an executable
```bash
gcc -o main.exe main.c static.a
```

### Example (dynamic lib)


Compile from source code `fn1.cpp` and `fn2.cpp`:
```bash
gcc -c fn1.cpp fn2.cpp
```
which produces `fn1.o` and `fn2.o`.

Get the shared/dynamic lib
```bash
gcc -shared -W -o libshared.so fn1.o fn2.o
```

Export the dir path where `libshared.so` resides, then output the executable `main.exe`.
``` bash
export LD_LIBRARY_PATH=/path/to/libshared:$LD_LIBRARY_PATH
gcc -Wall -o main.exe main.c -L/path/to/libshared -lshared
```

## Executable Entry

### ABI

AN application binary interface (ABI) is an interface between two binaries. For executable entry, it refers to how an executable has it machine code loaded by means of operating system into execution.

For Unix, here defines Executable and Linkable Format (*ELF*, formerly named Extensible Linking Format), is a common standard file format for executable files, object code, shared libraries, and core dumps. 

### `main()` invocation

When an executable (of format ELF) is expected to run, OS first invokes a system call `exec()`, which is defined in an executable file's `_init`/`.init` section, then branches to `.start`/`_start`, which invokes `main()`.

From `include/linux/init.h`:

```cpp
/* You should add __init immediately before the function name, like:
 *
 * static void __init initme(int x, int y)
 * {
 *    extern int z; z = x * y;
 * }
 */
```