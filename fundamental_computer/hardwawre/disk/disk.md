# Disk Knowledge

## Disk Partition and Format

* Partition: the division of a physical hard drive into one or more logical storage units called partitions.
* Formatting: It is the process of creating a file system (setting up the necessary file structures) on a partition.

### Typical Formats

1. FAT (used by Windows and Linux)

*File Allocation Table* (FAT) is a file system developed for personal computers. The file system uses an index table stored on the device to identify chains of data storage areas associated with a file.

The maximal possible size for a file on a FAT32 volume is 4 GB minus 1 byte, or 4,294,967,295 $(2^{32} − 1)$ bytes. This limit is a consequence of the 4-byte file length entry in the directory table.

exFAT extends file size to $(2^{64} − 1)$.

2. NTFS (used by Windows)

*New Technology File System* (NTFS) is a proprietary journaling file system (keep track of changes not yet committed to the file system's main part by recording the goal of such changes in a data structure known as a "journal") developed by Microsoft.

In computer file systems, a cluster (sometimes also called allocation unit or block) is a unit of disk space allocation for files and directories. The maximum NTFS volume size is $2^{64} − 1$ clusters, of which max cluster size is 2 MB.

1. ext4 (used by Linux)

*ext4* journaling file system or fourth extended filesystem is a journaling file system for Linux. 

It supports volumes with sizes up to 1 exbibyte (EiB) and single files with sizes up to 16 tebibytes (TiB) with the standard 4 KiB block size.

## Disk Interface Specifications

* SATA (Serial AT Attachment)

SATA (Serial AT Attachment) is a computer bus interface that connects host bus adapters to hard disks such as HDDs or SSDs.

* PCIe

PCI Express (Peripheral Component Interconnect Express) is a high-speed serial computer expansion bus standard.

* NVMe

NVM Express (NVMe) or Non-Volatile Memory Host Controller Interface Specification (NVMHCIS) is a specification to access computer's non-volatile storage media (aka hard disk) usually attached via the PCI Express bus.

### history of SATA, PCI Express and NVMe

Historically, SATA was designed for hard disk drives (HDDs).

Since SSDs became widely available, PCIe has been the popular equivalent of SATA.

NVMe was later proposed as a more advanced equivalent solution of PCIe dedicated to SSDs.

### NvLink vs PCIe

