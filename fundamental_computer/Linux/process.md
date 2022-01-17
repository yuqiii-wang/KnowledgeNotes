# Process

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

### Service vs Systemctl

`Service` is an "high-level" command used for start, restart, stop and status services in different Unixes and Linuxes, operating on the files in `/etc/init.d`.

`systemctl` operates on the files in `/lib/systemd`.

`service` is a **wrapper** for all three init systems (/init, systemd and upstart).

* Mask/Unmask a service

We should mask a service, if we want to prevent any kind of activation, even manual. e.g. If we don’t want to apply firewall rules at all then we can mask the `firewalld` service.

```bash
systemctl unmask firewalld
systemctl start firewalld
```

### Create a bootable usb

```bash
sudo umount /dev/sda1
sudo dd if=/path/to/ubuntu.iso of=/dev/sda1 bs=1M
```

### Kernel Levels:

A `runlevel` can simply be thought of as the state your system enters like if a system is in a single-user mode it will have a runlevel 1 while if the system is in a multi-user mode it will have a runlevel 5.

LINUX kernel supports these seven different runlevels :

0 – System halt i.e the system can be safely powered off with no activity.

1 – Single user mode.

2 – Multiple user mode with no NFS(network file system).

3 – Multiple user mode under the command line interface and not under the graphical user interface.

4 – User-definable.

5 – Multiple user mode under GUI (graphical user interface) and this is the standard runlevel for most of the LINUX based systems.

6 – Reboot which is used to restart the system.