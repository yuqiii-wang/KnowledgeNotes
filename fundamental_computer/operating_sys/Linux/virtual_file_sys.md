# Virtual File System

Linux treats external devices/peripherals (e.g. Disk, Flash, I/O, Display Monitor) as abstract files so that they can be managed by some unified file operations such as`open`、`close`、`read`、`write`.

The virtual file system is some like the Registry in Windows. It manages the OS configs as well.

## `/proc` and `sys`

There is no real file system exists on `/proc` or `/sys`, but virtual files residing in RAM that helps manage OS config. 

### `/proc`

|File name|Description|
|-|-|
|`/proc/cpuinfo`|	Information about CPUs in the system.
|`/proc/meminfo`|	information about memory usage.
|`/proc/ioports`|	list of port regions used for I/O communication with devices.
|`/proc/mdstat`|	display the status of RAID disks configuration.
|`/proc/kcore`|	displays the actual system memory.
|`/proc/modules`|	displays a list of kernel loaded modules.
|`/proc/cmdline`|	displays the passed boot parameters.
|`/proc/swaps`|	displays the status of swap partitions.
|`/proc/iomem`|	the current map of the system memory for each physical device.
|`/proc/version`|	displays the kernel version and time of compilation.
### `/sys`

`/sys` can be used to get information about your system hardware.

|File/Directory name|Description|
|-|-|
|Block|	list of block devices detected on the system like sda.
|Bus|	contains subdirectories for physical buses detected in the kernel.
|Class|	describes the class of device like audio, network, or printer.
|Devices|	list all detected devices by the physical bus registered with the kernel.
|Module|	lists all loaded modules.
|Power|	the power state of your devices.

### Use example

If you have multiple network cards, you can run below to config computer to use a particular card
```bash
echo "1" > /proc/sys/net/ipv4/ip_forward
``` 

To persist the change, you can either
```bash
sysctl -w net.ipv4.ip_forward=1
```
or
```bash
echo "net.ipv4.ip_forward = 1" >> /etc/sysctl.conf
```