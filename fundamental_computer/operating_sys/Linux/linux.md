# Linux OS

## Linux Components

![linux_components](imgs/linux_components.png "linux_components")

###  Kernel

Maintaining all the important abstractions of the operating system

Kernel refers to the privileged mode with full access to all the physical resources

* Kernel Modules

Kernel loads binary from disk into memory and maintains a symbol look-up table for object reference.

* Driver Registration

Common drivers are
1) device driver: printers/terminals and mice, disk
2) File systems
3) Network protocols
4) binaries: executables

* System Libraries

System libraries define a standard set of functions through which applications can interact with the kernel.

* System Utilities

System utilities are programs that perform individual, specialized management tasks, others known as daemons in UNIX terminology - may run permanently

## Linux process management

Linux manages processes with assigned priority and locks to shared memory access.

System calls are

* `fork()` creates a new process by duplicating the calling process. On success, the PID of the child process is returned in the parent, and 0 is returned in the child.

* `exec()` family of functions replaces the current process image with a new process image. It loads the program into the current process space and runs it from the entry point, such as `exec("ls")` runs `ls` from the current process.

* `clone()` gives a new process or a new thread depending on passed arguments to determined various shared memory regions. For example, `CLONE_FS` dictates shared file system; `CLONE_SIGHAND` dictates shared signal handlers. If with no argument flags, it is same as `fork()`.


## Linux Distributions

RedHat, SUSE, Fedora, Debian, Slackware, and Ubuntu.

`DEB` files are installation files for Debian based distributions. `RPM` files are installation files for Red Hat based distributions. Ubuntu is based on Debian’s package manage based on `APT` (apt-get repos) and `DPKG`. Red Hat, CentOS and Fedora are based on the old Red Hat Linux package management system, `RPM`.

### Yum Repos

GNU Privacy Guard (GnuPG or GPG) is a hybrid-encryption software program.

`rpm` utility uses GPG keys to sign packages and its own collection of imported public keys to verify packages. `yum` and `dnf` (successor to `yum`) use repository configuration files to provide pointers to the GPG public key locations and assist in importing the keys so that RPM can verify the packages.

`yum` uses /etc/yum.repos.d to specify a URL for the GPG key used to verify packages in that repository. 

use `rpm -K \path\to\rpm\file` to verify if key digest is valid.