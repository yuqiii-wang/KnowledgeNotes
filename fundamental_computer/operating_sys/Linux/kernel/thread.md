# Thread

Threads are created the same as normal tasks, with the exception that the `clone()` system call is passed flags corresponding to the specific resources to be shared:
```cpp
clone(CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND, 0);
```

The previous code results in behavior identical to a normal fork() , except that the address space, filesystem resources, file descriptors, and signal handlers are shared. 

## Kernel threads

Kernel threads do not have visible address, since they exist in kernel space that is invisible from user space.

A kernel thread can be created via `kthread_create` and made runnable by `kthread_run`.