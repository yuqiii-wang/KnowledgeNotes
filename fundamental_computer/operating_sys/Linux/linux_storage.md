# Linux Storage

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

An *inode* is a data structure that stores various information about a file in Linux, such as permisions (read, write, exe), file size, ownership, etc.

Each inode is identified by an integer number. An inode is assigned to a file when it is created.

use `ls -il` to show inode number (first col).

### Stream

Unix *stream* enables an application to assemble pipelines of driver code dynamically.