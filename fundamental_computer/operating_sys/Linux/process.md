# Process

## Linux process management

Linux manages processes with assigned priority and locks to shared memory access.

System calls are

* `fork()` creates a new process by duplicating the calling process. On success, the PID of the child process is returned in the parent, and 0 is returned in the child.

* `exec()` family of functions replaces the current process image with a new process image. It loads the program into the current process space and runs it from the entry point, such as `exec("ls")` runs `ls` from the current process.

* `clone()` gives a new process or a new thread depending on passed arguments to determined various shared memory regions. For example, `CLONE_FS` dictates shared file system; `CLONE_SIGHAND` dictates shared signal handlers. If with no argument flags, it is same as `fork()`.

* PID - Process ID

Process ID is a unique identifier of a process.

* PPID - Parent Process ID

The parent process ID of a process is the process ID of its creator, for the lifetime of the creator. After the creator's lifetime has ended, the parent process ID is the process ID of an implementation-defined system process.

* SID - Session ID

A collection of process groups established for job control purposes. Each process group is a member of a session.

* PGID - Process Group ID

A collection of processes that permits the signaling of related processes.

* EUID - Effective User ID

An attribute of a process that is used in determining various permissions, including file access permissions; see also User ID.

## Service vs Systemctl

`Service` is an "high-level" command used for start, restart, stop and status services in different Unixes and Linuxes, operating on the files in `/etc/init.d`.

`systemctl` operates on the files in `/lib/systemd`.

`service` is a **wrapper** for all three init systems (/init, systemd and upstart).

* Mask/Unmask a service

We should mask a service, if we want to prevent any kind of activation, even manual. e.g. If we don’t want to apply firewall rules at all then we can mask the `firewalld` service.

```bash
systemctl unmask firewalld
systemctl start firewalld
```

## Create a bootable usb

```bash
sudo umount /dev/sda1
sudo dd if=/path/to/ubuntu.iso of=/dev/sda1 bs=1M
```

