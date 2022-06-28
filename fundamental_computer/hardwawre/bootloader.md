# Bootloader when a Computer Starts

## When a Computer is Powered on

1. You press a power button

2. CPU defers to load instructions from the firmware chip on the motherboard and begins executing instructions.

3. The firmware code does a Power On Self Test (POST), initializing and testing hardware

4. Finally, the firmware code cycles through all storage devices and looks for a boot-loader (usually located in first sector of a disk). If the boot-loader is found, then the firmware hands over control of the computer to it.

5. bootloader loads the rest of OS (such as GRUB for unix-like OS).

6. bootloader loads kernel into memory, and run a master process (such as `init` for unix-like OS; `wininit.exe` and `services.exe` for windows)

7. Wait till all necessary hardware and drivers are ready, bootloader loads Graphical User Inferface (GUI)

## BIOS vs UEFI

### BIOS 

BIOS stands for Basic Input/Output System.

It is firmware embedded on the chip on the computer's motherboard. BIOS firmware is pre-installed on the motherboard of a PC. It is a non-volatile firmware which means its settings won’t disappear or change even after power off.

### UEFI

UEFI stands for Unified Extensible Firmware Interface.

There is disk format requirement of booting with UEFI.

UEFI extends BIOS's services in many ways, such as:

* If your disk is larger than 2TB partition, you should use UEFI

* UEFI can have GUI (capable of navigation via mouse click)

* UEFI has faster booting time

* UEFI supports secure startup, checking malware during the startup process