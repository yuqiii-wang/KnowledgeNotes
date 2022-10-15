# Memory for User Processes

The kernel represents a process’s address space with a data structure called the *memory
descriptor*. This structure contains all the information related to the process address space.
The memory descriptor is represented by `struct mm_struct`.

This memory structure defines where to store a process's code, data, text, etc.
```cpp
struct mm_struct {
    struct vm_area_struct       *mmap;      /* list of memory areas */
    struct rb_root              mm_rb;      /* red-black tree of VMAs */
    struct vm_area_struct       *mmap_cache;/* last used memory area */
    unsigned long               free_area_cache;/* 1st address space hole */
    pgd_t                       *pgd;       /* page global directory */
    atomic_t                    mm_users;   /* address space users */
    spinlock_t                  page_table_lock;/* page table lock */
    struct list_head            mmlist;     /* list of all mm_structs */
    unsigned long               start_code; /* start address of code */
    unsigned long               end_code;   /* final address of code */
    unsigned long               start_data; /* start address of data */
    unsigned long               end_data;   /* final address of data */
    unsigned long               start_brk;  /* start address of heap */
    unsigned long               brk;        /* final address of heap */
    unsigned long               start_stack;/* start address of stack */
    unsigned long               arg_start;  /* start of arguments */
    unsigned long               arg_end;    /* end of arguments */
    unsigned long               env_start;  /* start of environment */
    unsigned long               env_end;    /* end of environment */
    unsigned long               total_vm;   /* total number of pages */
    unsigned long               locked_vm;  /* number of locked pages */
};
```
where 

* the `mm_users` field is the number of processes using this address space. For example, if two threads share this address space, `mm_users` is equal to two
* the `mm_count` field is the primary reference count for the `mm_struct`.

Segmentation fault refers to a process accessing a memory address not in a valid memory area, or if it accesses a valid area in an invalid manner, the kernel kills the process.


## Virtual Memory

The `vm_area_struct` structure describes a single memory area over a contiguous interval in a given address space. The kernel treats each memory area as a unique memory object. Each memory area possesses certain properties, such as permissions and a set of associated operations.

```cpp
struct vm_area_struct {
    struct mm_struct        *vm_mm;     /* associated mm_struct */
    unsigned long           vm_start;   /* VMA start, inclusive */
    unsigned long           vm_end;     /* VMA end , exclusive */
    struct vm_area_struct   *vm_next;   /* list of VMA’s */
    pgprot_t                vm_page_prot; /* access permissions */
    unsigned long           vm_flags;   /* flags */
    struct rb_node          vm_rb;      /* VMA’s node in the tree */
    union {                             /* links to address_space->i_mmap or i_mmap_nonlinear */
        struct {
            struct list_head        list;
            void                    *parent;
            struct vm_area_struct   *head;
        } vm_set;
        struct prio_tree_node prio_tree_node;
    } shared;
    struct list_head        anon_vma_node;  /* anon_vma entry */
    struct anon_vma         *anon_vma;      /* anonymous VMA object */
    struct vm_operations_struct *vm_ops;    /* associated ops */
    unsigned long           vm_pgoff;       /* offset within file */
    struct file             *vm_file;       /* mapped file, if any */
    void                    *vm_private_data; /* private data */
};
```
where
* The `vm_start` field is the initial (lowest) address in the interval,
* The `vm_end` field is the first byte after the final (highest) address in the interval.
* `vm_end` - `vm_start` is the length in bytes of the memory area.
* The `mmap` and `mm_rb` fields are different data structures that contain the same thing: all the memory areas in this address space. The redundancy of having two data structures storing the same gives the benefits: a linked list, allows for simple and efficient traversing of all elements; a red-black tree, is more suitable to searching for a given element.
* All of the `mm_struct` structures are strung together in a doubly linked list via the `mmlist` field.

If two processes map the same file into their respective address spaces, each has a unique `vm_area_struct` to identify its unique memory area. Conversely, two threads that share an address space also share all the `vm_area_struct` structures therein.

### Virtual Memory Flags

|Flags|Comments|
|-|-|
|VM_READ|Pages can be read from|
|VM_WRITE|Pages can be written to|
|VM_EXEC|Pages can be executed|
|VM_SHARED|Pages are shared|
|VM_SHM|The area is used for shared memory (among multiple processes); if not set, the mapped memory is private|
|VM_DENYWRITE|The area maps an unwritable file|
|VM_EXECUTABLE|The area maps an executable file|
|VM_LOCKED|The pages in this area are locked|
|VM_IO|The area maps a device’s I/O space|
|VM_SEQ_READ|The pages seem to be accessed sequentially; the kernel can then opt to increase the read-ahead performed on the backing file.|
|VM_RAND_READ|The pages seem to be accessed randomly|
|VM_RESERVED|This area must not be swapped out|

### VMA Operations

```cpp
struct vm_operations_struct {
    void (*open) (struct vm_area_struct *);
    void (*close) (struct vm_area_struct *);
    int (*fault) (struct vm_area_struct *, struct vm_fault *);
    int (*page_mkwrite) (struct vm_area_struct *vma, struct vm_fault *vmf);
    int (*access) (struct vm_area_struct *, unsigned long , void *, int, int);
};
```
where
* `open` This function is invoked when the given memory area is added to an address space.
* `close` This function is invoked when the given memory area is removed from an address memory
* `fault` This function is invoked by the page fault handler when a page that is not present in physical memory is accessed.

### Example

Process virtual memory mapping can be checked via `/proc/<pid>/maps`.

For example, given this program
```cpp
#include <unistd.h>

int main(int argc, char *argv[])
{
    sleep(100);
    return 0;
}
```
whose `/proc/<pid>/maps` output is
```bash
55be260d3000-55be260d4000 r--p 00000000 103:05 2365187                   /home/yuqi/Desktop/KnowledgeNotes/fundamental_computer/operating_sys/Linux/kernel/a.out
55be260d4000-55be260d5000 r-xp 00001000 103:05 2365187                   /home/yuqi/Desktop/KnowledgeNotes/fundamental_computer/operating_sys/Linux/kernel/a.out
55be260d5000-55be260d6000 r--p 00002000 103:05 2365187                   /home/yuqi/Desktop/KnowledgeNotes/fundamental_computer/operating_sys/Linux/kernel/a.out
55be260d6000-55be260d7000 r--p 00002000 103:05 2365187                   /home/yuqi/Desktop/KnowledgeNotes/fundamental_computer/operating_sys/Linux/kernel/a.out
55be260d7000-55be260d8000 rw-p 00003000 103:05 2365187                   /home/yuqi/Desktop/KnowledgeNotes/fundamental_computer/operating_sys/Linux/kernel/a.out
7f49f6f35000-7f49f6f57000 r--p 00000000 103:05 3672761                   /usr/lib/x86_64-linux-gnu/libc-2.31.so
7f49f6f57000-7f49f70cf000 r-xp 00022000 103:05 3672761                   /usr/lib/x86_64-linux-gnu/libc-2.31.so
7f49f70cf000-7f49f711d000 r--p 0019a000 103:05 3672761                   /usr/lib/x86_64-linux-gnu/libc-2.31.so
7f49f711d000-7f49f7121000 r--p 001e7000 103:05 3672761                   /usr/lib/x86_64-linux-gnu/libc-2.31.so
7f49f7121000-7f49f7123000 rw-p 001eb000 103:05 3672761                   /usr/lib/x86_64-linux-gnu/libc-2.31.so
7f49f7123000-7f49f7129000 rw-p 00000000 00:00 0 
7f49f716c000-7f49f716d000 r--p 00000000 103:05 3672234                   /usr/lib/x86_64-linux-gnu/ld-2.31.so
7f49f716d000-7f49f7190000 r-xp 00001000 103:05 3672234                   /usr/lib/x86_64-linux-gnu/ld-2.31.so
7f49f7190000-7f49f7198000 r--p 00024000 103:05 3672234                   /usr/lib/x86_64-linux-gnu/ld-2.31.so
7f49f7199000-7f49f719a000 r--p 0002c000 103:05 3672234                   /usr/lib/x86_64-linux-gnu/ld-2.31.so
7f49f719a000-7f49f719b000 rw-p 0002d000 103:05 3672234                   /usr/lib/x86_64-linux-gnu/ld-2.31.so
7f49f719b000-7f49f719c000 rw-p 00000000 00:00 0 
7ffcd1e96000-7ffcd1eb8000 rw-p 00000000 00:00 0                          [stack]
7ffcd1f58000-7ffcd1f5c000 r--p 00000000 00:00 0                          [vvar]
7ffcd1f5c000-7ffcd1f5e000 r-xp 00000000 00:00 0                          [vdso]
ffffffffff600000-ffffffffff601000 --xp 00000000 00:00 0                  [vsyscall]
```

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