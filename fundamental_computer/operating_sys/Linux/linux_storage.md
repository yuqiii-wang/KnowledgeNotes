# Linux Storage

## Files

|Symbol|	Meaning|
|-|-|
|-|	Regular file
|d|	Directory
|l|	Link
|c|	Special file
|s|	Socket
|p|	Named pipe
|b|	Block device
|x| Executable

### Partitions

* data partition: normal Linux system data, including the root partition containing all the data to start up and run the system; and

* swap partition: expansion of the computer's physical memory, extra memory on hard disk.

Most Linux systems use `fdisk` at installation time to set the partition type. The standard Linux partitions have number 82 for swap and 83 for data, which can be journaled (ext3) or normal (ext2, on older systems), besides other file system types such as JFS, NFS, FATxx. 

## Typical root directories

|Directory|Description|
|-|-|
|/bin|	The /bin directory contains user executable files.
|/boot|	Contains the static bootloader and kernel executable and configuration files required to boot a Linux computer.
|/dev|	This directory contains the device files for every hardware device attached to the system. These are not device drivers, rather they are files that represent each device on the computer and facilitate access to those devices.
|/etc|	Contains the local system configuration files for the host computer.
|/home|	Home directory storage for user files. Each user has a subdirectory in /home.
|/lib|	Contains shared library files that are required to boot the system.
|/media|	A place to mount external removable media devices such as USB thumb drives that may be connected to the host.
|/mnt|	A temporary mountpoint for regular filesystems (as in not removable media) that can be used while the administrator is repairing or working on a filesystem.
|/opt|	Optional files such as vendor supplied application programs should be located here.
|/root|	This is not the root (/) filesystem. It is the home directory for the root user.
|/sbin|	System binary files. These are executables used for system administration.
|/tmp|	Temporary directory. Used by the operating system and many programs to store temporary files. Users may also store files here temporarily. Note that files stored here may be deleted at any time without prior notice.
|/usr|	These are shareable, read-only files, including executable binaries and libraries, man files, and other types of documentation.
|/var|	Variable data files are stored here. This can include things like log files, MySQL, and other database files, web server data files, email inboxes, and much more.

## Devices

Devices are in `/dev` directory.

Block Devices tend to be storage devices, capable of buffering output and storing data for later retrieval.

Character Devices are things like audio or graphics cards, or input devices like keyboard and mouse.

### Loop devices

The loop device is a block device that maps its data blocks not
to a physical device such as a hard disk or optical disk drive,
but to the blocks of a regular file in a filesystem or to another
block device. 

## Logical Volume Manager (LVM)

Logical Volume Manager (LVM) is a device mapper framework that provides logical volume management. Available cmds for use:
```bash
sudo apt-get install lvm2
```

Flexible in contrast to partitioning that treats disk as separate regions.

Terminologies:
* PV: Physical Volume (lowest layer of LVM, on top of partitions)
* PE: Physical Extents (equal sized segements of PV)
* VG: Volume Group (storage pool made up of PVs)
* LV: Logical Volume (created from free PEs)

### Snapshot

File system point-in-time view backup.

Copy-on-write to monitor changes to existing data blocks.

## Inode

An *inode* is a data structure that stores various information about a file in Linux, such as permissions (read, write, exe), file size, ownership, etc.

Each inode is identified by an integer number. An inode is assigned to a file when it is created.

use `ls -il` to show inode number (first col).

### Stream

Unix *stream* enables an application to assemble pipelines of driver code dynamically.