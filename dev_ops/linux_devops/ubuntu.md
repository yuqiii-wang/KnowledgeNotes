# Ubuntu

By `uname -r`, ubuntu uses `Linux 5.15.0-91-generic`.


## GRUB

GNU GRUB is a Multiboot boot loader. It was derived from GRUB, the *GRand Unified Bootloader*.

## GUI and GNOME

### Non-GUI Login

Switch to non-GUI mode after reboot: `Ctrl`-`Alt`-`F1` 
https://superuser.com/questions/100693/how-to-switch-to-non-graphical-view-in-ubuntu

By default, recent Ubuntu releases (by the year 2023) have virtual terminal 1 - 6 corresponding to `Ctrl`-`Alt`-`F1` to `Ctrl`-`Alt`-`F6`.
The 7th is graphical entry.

### X.Org

X.Org Server is the free and open-source implementation of the X Window System (X11) display server stewarded by the X.Org Foundation.

The X Window System (X11, or simply X) is a windowing system for bitmap displays, common on Unix-like operating systems.

X provides the basic framework for a GUI environment: drawing and moving windows on the display device and interacting with a mouse and keyboard.

### OpenGL Select

`glxinfo | grep vendor` can see what GPU is used for OpenGL tasks.

Most likely would get `Intel` for modern Intel CPU has builtin mini GPU.

To switch to use Nvidia, first run `sudo prime-select query` to check if "nvidia"-like texts are output,
if not, run `sudo prime-select nvidia` to select OpenGL renderer as Nvidia.

Reboot, then run again `glxinfo | grep vendor` to check if it shows "Nvidia"-like texts.