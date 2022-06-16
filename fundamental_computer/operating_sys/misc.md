# Some Miscellaneous Knowledge

`/proc/kcore`

This  file  represents  the physical memory of the system and is stored in the ELF core file format.  With this pseudo-file, 
and an unstripped kernel (/usr/src/linux/vmlinux) binary, 
GDB can be used to examine the current state of any kernel data structures.

The total length of the file is  the  size  of  physical  memory
(RAM) plus 4KB.