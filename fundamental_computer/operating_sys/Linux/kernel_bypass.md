## Kernel bypass

Linux has two separate spaces: user space and kernel space. For example, in Linux socket, tcp packets received from NIC first arrives at kernel space then copied to user space. Kernel bypass means no engagement of kernel operations but directly retrieving data from NIC to user space. 

Besides TCP handling, system calls without privileged execution all exposed in the user space are use cases exemplifying kernel bypass.

## Zero copy

"Zero-copy" describes computer operations in which the CPU does not perform the task of copying data from one memory area to another or in which unnecessary data copies are avoided.

Zero-copy programming techniques can be used when exchanging data within a user space process (i.e. between two or more threads, etc.) and/or between two or more processes (see also producer–consumer problem) and/or when data has to be accessed / copied / moved inside kernel space or between a user space process and kernel space portions of operating systems (OS).