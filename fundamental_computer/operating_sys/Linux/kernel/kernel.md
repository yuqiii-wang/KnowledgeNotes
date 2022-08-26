# Linux Kernel

## Kernel Levels:

A `runlevel` can simply be thought of as the state your system enters like if a system is in a single-user mode it will have a runlevel 1 while if the system is in a multi-user mode it will have a runlevel 5.

LINUX kernel supports these seven different runlevels :

0 – System halt i.e the system can be safely powered off with no activity.

1 – Single user mode.

2 – Multiple user mode with no NFS(network file system).

3 – Multiple user mode under the command line interface and not under the graphical user interface.

4 – User-definable.

5 – Multiple user mode under GUI (graphical user interface) and this is the standard runlevel for most of the LINUX based systems.

6 – Reboot which is used to restart the system.


## Terminal

### TTY

**tty** (TeleTYpewriter) is a native terminal device; the backend is either hardware or kernel emulated.

* Historically speaking, user interacts with a terminal and communicate with computer through UART (Universal Asynchronous Receiver and Transmitter) which directs bytes to various processes.

![alt text](imgs/tty.png "tty")

* Given fruther integrations, modern console does not have UART, emulated with software.

![alt text](imgs/tty_no_uart.png "tty_no_uart")

Use `alt + crtl + F1~F7` to switch between virtual terminals from `tty1` to `tty6`. `tty0` is root virtual terminal.

### pty

**pty** (pseudo-tty), a pair of slave (pts) and master (ptmx) provides  an  interface that behaves exactly like a classical terminal (tty).
