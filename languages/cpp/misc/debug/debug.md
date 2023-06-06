# Debug

## GDB

GBD can be used to debug c++ code, finding break point and display symbol tables.

### Quick Start

```bash
file <your-binary> # load symbol  table
run # execute
where # find failure point
info sharedlibrary # check the actually loaded libraries
```

Or directly from terminal (if `<args>` contains a dash `-`, use double quotes such as `-<args>` ):
```bash
gdb --args <your-binary> <arg1> <arg2> ...
```

### Debugging Symbol Table

A Debugging Symbol Table maps instructions in the compiled binary program to their corresponding variable, function, or line in the source code. 

Compile to render a symbol table by `-g` flag:
```cpp
gcc -g hello.cc -o hello 
```

Add a break point to code.

### Checking A Core Dump

1. `gdb path/to/the/binary path/to/the/core/dump/file`

2. `bt` (backtrace) to get the stack trace when the program crashed

3. `list` to see code around the function

4. `info locals` to check local variables

### Example: GDB Run Live

```cpp
#include <iostream>
using namespace std;  

int divint(int, int);  
int main() 
{ 
   int x = 5, y = 2; 
   cout << divint(x, y); 
   
   x =3; y = 0; 
   cout << divint(x, y); 
   
   return 0; 
}  

int divint(int a, int b) 
{ 
   return a / b; 
}  
```

Compile the code by `$g++ -g crash.cc -o crash`

then to go into the symbol table
```bash
gdb crash
```

run `r` to run a program inside gdb
```bash
r
```

run `where` to find at which line it fails.
```bash
where
```

### Use `launch.json` for vs code

Install GDB and config `launch.json`

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [

        {
            "name": "(gdb) Launch",
            "type": "cppdbg",
            "request": "launch",        
            "program": "${workspaceFolder}/build/crash",
            "args": ["arg1", "arg2"],
            "stopAtEntry": false,
            "cwd": "${fileDirname}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ]
        }
    ]
}
```

### GDB Run Executable with Args

* Syntax:
```bash
gdb --args executablename arg1 arg2 arg3
```


* Environment Vars: show, set and unset

Inside gdb, run
```bash
show environment 
set environment varname = foo
unset environment varname
```


## Valgrind

Valgrind is for memory debugging, memory leak detection, and profiling.
* Install by `sudo apt-get install valgrind`
* Run by `valgrind --tool=memcheck ./<program> --leak-check=full`
  
Alternatively, run with GDB (NOT recommended, as gdb may add symbols/change code block that `valgrind` may fail to detect mem leak)
* `valgrind --vgdb=yes --vgdb-error=0 <program> <arguments>` to init valgrind
* `gdb <program>` gdb run
* in gdb: `set non-stop off` 
* in gdb: `target remote | vgdb` gdb connects to valgrind
* in gdb: `monitor leak_check`

Example memory leak code:
```cpp
#include <iostream>

int main() {
    std::cout << "main\n";
    int * a = new int[10]; // mem leak for no `delete a;`
    return 0;
}
```

Valgrind should give the below report
```bash
==438518== 
==438518== HEAP SUMMARY:
==438518==     in use at exit: 40 bytes in 1 blocks
==438518==   total heap usage: 3 allocs, 2 frees, 73,768 bytes allocated
==438518== 
==438518== LEAK SUMMARY:
==438518==    definitely lost: 40 bytes in 1 blocks
==438518==    indirectly lost: 0 bytes in 0 blocks
==438518==      possibly lost: 0 bytes in 0 blocks
==438518==    still reachable: 0 bytes in 0 blocks
==438518==         suppressed: 0 bytes in 0 blocks
==438518== Rerun with --leak-check=full to see details of leaked memory
==438518== 
==438518== For lists of detected and suppressed errors, rerun with: -s
==438518== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
```

### Find Memory Leak Location

Need to compile with the flags 
`gcc -o <program> -std=c11 -Wall -ggdb3 <program>.cpp`, or in CMakeLIsts.txt:
```bash
target_compile_options(mem_leak_test_simple PRIVATE -Wall)
target_compile_options(mem_leak_test_simple PRIVATE -ggdb3)
```

Then run Valgrind:
```bash
valgrind --tool=memcheck \
         --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         --verbose \
         ./<program>
```
that should give the below report, where memory leak location is identified.
```bash
==442705== 
==442705== HEAP SUMMARY:
==442705==     in use at exit: 40 bytes in 1 blocks
==442705==   total heap usage: 3 allocs, 2 frees, 73,768 bytes allocated
==442705== 
==442705== Searching for pointers to 1 not-freed blocks
==442705== Checked 113,496 bytes
==442705== 
==442705== 40 bytes in 1 blocks are definitely lost in loss record 1 of 1
==442705==    at 0x4A3AAAF: operator new[](unsigned long) (vg_replace_malloc.c:652)
==442705==    by 0x1091D1: main (mem_leak_test_simple.cpp:5)
==442705== 
==442705== LEAK SUMMARY:
==442705==    definitely lost: 40 bytes in 1 blocks
==442705==    indirectly lost: 0 bytes in 0 blocks
==442705==      possibly lost: 0 bytes in 0 blocks
==442705==    still reachable: 0 bytes in 0 blocks
==442705==         suppressed: 0 bytes in 0 blocks
==442705== 
==442705== ERROR SUMMARY: 1 errors from 1 contexts (suppressed: 0 from 0)
```

## Address Sanitizer

Address sanitizer can check if there exist array boundary breach.

`g++` has already included this tool.
Compile program with this flag should be good `-fsanitize=address`