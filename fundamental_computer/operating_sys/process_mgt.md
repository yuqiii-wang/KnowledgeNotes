# Process and thread management

Each process is represented in OS as a process control block. Most typical information is
* process state (new, ready, waiting, running, terminated)
* process number (PID)
* program counter (a dedicated CPU register to record next instruction for execution)
* CPU scheduling
* Memory management
* accounting (used memory, cpu, etc.)

## Process scheduling

A process scheduler manages executions of programs to optimize the use of CPU and memory, such as managing process context switch (push/pop registers and memory addrs).

Use `ps -elf` to list currently running processes on an OS. Each PID is forked from PPID (Parent PID). 

* PID 0 swapper/sched: scheduler, memory paging, to load data from secondary memory (disk) into main memory (RAM) 
* PID 1 init: process primarily responsible for starting and shutting down the system. 
* PID 2 kthreadd: is the kernel thread daemon. All kthreads are forked from this thread. 

### CPU scheduler

Whenever the CPU becomes idle, the operating system must select one of the
processes in the ready queue to be executed. 
The selection process is carried out by the short-term scheduler (or CPU scheduler). The queued processes are selected (expected to be executed) by various best performance scheduling algorithms such as priority order.

* nonpreemptive or cooperative: process be terminated or moved to wait state.

* preemptive: a process switches from the waiting state to the ready state (for example, at completion of I/0) or from running state to wait state (e.g., system interrupt)

A preemptive scheduler will allow a particular process to run for a short amount of time called a *quantum* (or time slice). After this amount of time, the process is placed back in the ready queue and another process is placed into the run state.

### Linux Scheduling

Linux scheduler is a preemptive, priority-based algorithm with two separate priority ranges: a real-time range from $0$ to $99$ and a nice value ranging from $100$ to $140$:

![linux_scheduling](imgs/linux_scheduling.png "linux_scheduling")

Synchronizations of threads' code are handled by `mutex`/`semaphore`.

## Process communications

Processes communicate to each other via

* `send`/`recv` message passing 
* declare a shared memory.
```cpp
/* allocate a shared memory segment */
segment_id = shmget(IPC_PRIVATE, size, s_IRUSR I s_IWUSR);
/* attach the shared memory segment */
shared_memory = (char*) shmat(segment_id, NULL, 0);
/* write a message to the shared memory segment */
sprintf (shared_memory, "Hi there!");
```

* use socket, such as unix socket by TCP/IP

* unix pipe

UNIX treats a pipe as a special type of file for data streaming
```cpp
pipe (int fd []) // fd: file descriptor
```

On bash, `|` is a pipe communication directing data from previous process to next's.

## Thread

A thread is a basic unit of CPU utilization; it comprises a thread ID, a program
counter, a register set, and a stack.

Thread has two parts: user threads and kernel threads; One program typically have both. 

For Unix, `Pthread` refers to the POSIX standard (IEEE 1003.lc) defining an API for thread
creation and synchronization.

It depends on OS to implement `fork()` from a parent process, whether all threads from a parent process got duplicated, or just  the main thread.

## Parallelism and Sync

A collection of instructions (or operations) that performs a single logical
function is called a *transaction*. *Atomic transaction* refers to the block of logic/data being executed by only one thread. When 

*Deadlock* happens when a lock $LA$ check waits for another lock ($LB$)'s release, while $LB$ waits for $LA$'s release. It can be remedied by

1) drawing a lock-dependency graph and avoid deadlock in advance, 
2) terminate a process releasing all locks simultaneously, 
3) starvation that forces release/rollback a section of code/state after a period of time
