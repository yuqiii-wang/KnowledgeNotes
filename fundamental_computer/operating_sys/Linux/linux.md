# Linux OS

## Linux Components

![linux_components](imgs/linux_components.png "linux_components")

### Kernel

Kernel has below functionalities

* Process Scheduler

* Memory Manager

* VFS (Virtual File System)

* Network

* IPC (Inter-Process Communication)

## Linux Distributions

RedHat, SUSE, Fedora, Debian, Slackware, and Ubuntu.

`DEB` files are installation files for Debian based distributions. `RPM` files are installation files for Red Hat based distributions. Ubuntu is based on Debian's package manage based on `APT` (apt-get repos) and `DPKG`. Red Hat, CentOS and Fedora are based on the old Red Hat Linux package management system, `RPM`.

### Yum Repos

GNU Privacy Guard (GnuPG or GPG) is a hybrid-encryption software program.

`rpm` utility uses GPG keys to sign packages and its own collection of imported public keys to verify packages. `yum` and `dnf` (successor to `yum`) use repository configuration files to provide pointers to the GPG public key locations and assist in importing the keys so that RPM can verify the packages.

`yum` uses /etc/yum.repos.d to specify a URL for the GPG key used to verify packages in that repository. 

use `rpm -K \path\to\rpm\file` to verify if key digest is valid.

## Linux Is NOT Realtime

A realtime OS mandates that a task must be completed before a deadline, even taking priority over other tasks.

This is done by permitting higher-level priority interrupt handler triggered at anytime during another process's run.

For example, 

## The `glibc`

The GNU C Library, commonly known as *glibc*, is an implementation of the C standard library most widely used in the Linux ecosystem.

`glibc` strictly follows industry standards such as POSIX, ISO C, and others, ensuring compatibility with software developed for Unix-like systems.

* System Call Wrappers:

Interfaces to Linux kernel system calls such as file operations (`open`, `read`, `write`), process management (`fork`, `exec`), and inter-process communication.

* Memory Management:

Functions like `malloc`, `free`, `calloc`, and `realloc` for dynamic memory allocation.

* String Manipulation:

Utilities like `strlen`, `strcpy`, `strcat`, and `strcmp` for handling C-style strings.

* I/O Operations:

Functions like `printf`, `scanf`, `fopen`, `fclose`, `fread`, and `fwrite` for standard input/output and file handling.

* Time and Date:

Functions like `time`, `clock`, `strftime`, and `localtime` for managing and formatting dates and times.

* Threading:

Support for multi-threaded programming through the POSIX threads (`pthreads`) API.

* Networking:

Functions for socket programming and network communications, such as `socket`, `bind`, `connect`, `send`, and `recv`.

* Mathematics:

A wide range of mathematical functions, including `sin`, `cos`, `sqrt`, and `log`.

* Process and Environment Management:

Functions like `getpid`, `getenv`, `setenv`, and `execvp` for managing processes and environment variables.
