# Disk Knowledge

## Disk Format

To list file system info
```bash
lsblk -f
```

### Format Examples

To format a disk to ext4, where `/dev/sda1` is the usb name searchable from `lsblk -f`.
```bash
sudo mkfs -t ext4 /dev/sda1
```

To format `exfat` on Linux

```bash
sudo apt install exfat-utils
sudo mkfs.exfat -n exfat_usb /dev/sda1
```

1. FAT

File Allocation Table (FAT) is a file system developed for personal computers. The file system uses an index table stored on the device to identify chains of data storage areas associated with a file.

The maximal possible size for a file on a FAT32 volume is 4 GB minus 1 byte, or 4,294,967,295 $(2^{32} − 1)$ bytes. This limit is a consequence of the 4-byte file length entry in the directory table.

exFAT extends file size to $(2^{64} − 1)$.

2. NTFS

NTFS (New Technology File System) is a proprietary journaling file system (keep track of changes not yet committed to the file system's main part by recording the goal of such changes in a data structure known as a "journal") developed by Microsoft.

In computer file systems, a cluster (sometimes also called allocation unit or block) is a unit of disk space allocation for files and directories. The maximum NTFS volume size is $2^{64} − 1$ clusters, of which max cluster size is 2 MB.

3. ext4

ext4 journaling file system or fourth extended filesystem is a journaling file system for Linux. 

It supports volumes with sizes up to 1 exbibyte (EiB) and single files with sizes up to 16 tebibytes (TiB) with the standard 4 KiB block size.