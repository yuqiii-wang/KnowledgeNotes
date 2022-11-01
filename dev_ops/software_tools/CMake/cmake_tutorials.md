# cmake

## bash to build
```bash
mkdir build
cd build
cmake ../build
cmake --build .
```

## cmake：target_** PUBLIC，PRIVATE，INTERFACE

```
cmake-test/                 Project Folder
├── CMakeLists.txt
├── hello-world             build libhello-world.so，contained hello_world.c and hello_world.h,
|   |                       and two sub-directories: hello and world
│   ├── CMakeLists.txt
│   ├── hello               build libhello.so 
│   │   ├── CMakeLists.txt
│   │   ├── hello.c
│   │   └── hello.h         
│   ├── hello_world.c
│   ├── hello_world.h       
│   └── world               build libworld.so
│       ├── CMakeLists.txt
│       ├── world.c
│       └── world.h         
└── main.c
```

Execution path:
```
                                    |---> libhello.so
executable ---> libhello-world.so ---
                                    |---> libworld.so
```

* `PUBLIC`

When declaring `hello-world` as `PRIVATE`, when building `libhello-world.so`, only `hello_world.c` has the include header files inside `hello.h`，`libhello-world.so`, whereas `hello_world.h` does not contain `hello.h`.

For `main.c` does not know `hello.c`, should write the below in `hello-world/CMakeLists.txt`

```bash
target_link_libraries(hello-world PRIVATE hello)
target_include_directories(hello-world PRIVATE hello)
```

* `INTERFACE`

When declaring `hello-world` as `INTERFACE`, when building `libhello-world.so`, only inside `hello_world.h` has `hello.h`, and `hello_world.c` does not contain `hello.h`.

Should write the below in `hello-world/CMakeLists.txt`
```bash
target_link_libraries(hello-world INTERFACE hello)
target_include_directories(hello-world INTERFACE hello)
```

* `PUBLIC`

Scope is equivalent to `PUBLIC` = `PRIVATE` + `INTERFACE`. When building `libhello-world.so`, both `hello_world.c` and `hello_world.h` have `hello.h`. In other words, `main.c` needs to access `libhello.so`.

Should write the below in `hello-world/CMakeLists.txt`

```bash
target_link_libraries(hello-world PUBLIC hello)
target_include_directories(hello-world PUBLIC hello) 
```