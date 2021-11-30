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