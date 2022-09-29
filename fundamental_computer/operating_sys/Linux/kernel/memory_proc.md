# Memory for User Processes

The kernel represents a process’s address space with a data structure called the *memory
descriptor*. This structure contains all the information related to the process address space.
The memory descriptor is represented by `struct mm_struct`.

## High/Low Memory

A 32-bit OS can only access a little less than $2^{32}$ ($4$ GB) bytes of virtual memory. 

### Mem Mapping

On a 32-bit architecture, the address space range for addressing RAM is:
```bash
0x00000000 - 0xffffffff
```

User space (low memory) would take
```bash
0x00000000 - 0xbfffffff
```
Every newly spawned user process gets an address (range) inside this area. User processes are generally untrusted and therefore are forbidden to access the kernel space. Further, they are considered non-urgent, as a general rule, the kernel tries to defer the allocation of memory to those processes.

The kernel space (high memory) range:
```bash
0xc0000000 - 0xffffffff
```
Processes spawned in kernel space are trusted, urgent and assumed error-free, the memory request gets processed instantaneously.

### Implementation

So when a 32-bit kernel needs to map more than $4$ GB of memory, it must be compiled with *high memory* support. High memory is memory which is not permanently mapped in the kernel's address space. (*Low memory* is the opposite: it is always mapped, so you can access it in the kernel simply by dereferencing a pointer).

When you access high memory from kernel code, you need to call `kmap` first, to obtain a pointer from a page data structure `struct page`. Calling `kmap` works whether the page is in high or low memory. The pointer obtained through `kmap` is a resource: it uses up address space. Once you've finished with it, you must call `kunmap` (or `kunmap_atomic`) to free that resource; then the pointer is no longer valid, and the contents of the page can't be accessed until you call `kmap` again.

## Process Stack

The size of the per-process kernel stacks depends on both the architecture and a compile-time option. Historically, the kernel stack has been two pages per process.This is usually 8KB for 32-bit architectures and 16KB for 64-bit architectures because they usually have 4KB and 8KB pages, respectively.

In any given function, you must keep stack usage to a minimum.

## Per-CPU Allocation

Modern SMP-capable operating systems use per-CPU data—data that is unique to a given processor—extensively. Typically, per-CPU data is stored in an array. Each item in the array corresponds to a possible processor on the system.

To get the CPU selected by the current process:
```cpp
int cpu = get_cpu();
```