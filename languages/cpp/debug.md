# Debug

## GDB

### Debugging Symbol Table

A Debugging Symbol Table maps instructions in the compiled binary program to their corresponding variable, function, or line in the source code. 

Compile to render a symbol table by `-g` flag:
```cpp
gcc -g hello.cc -o hello 
```

Add a break point to code 

### Example: A coredump

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

## Valgrind

Valgrind is for memory debugging, memory leak detection, and profiling.

