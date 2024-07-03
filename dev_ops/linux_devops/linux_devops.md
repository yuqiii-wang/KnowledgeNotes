# Some Linux Knowledge and DevOps

## `systemd`

`systemd` is a software suite that provides an array of system components for Linux, offered reliable parallelism during boot as well as centralized management of processes, daemons, services and mount points.
It provides management to computer resource allocation as well.

<div style="display: flex; justify-content: center;">
      <img src="imgs/systemd-and-cgroups.png" width="30%" height="50%" alt="systemd-and-cgroups" />
</div>
</br>

where *cgroups* (abbreviated from control groups) is a Linux kernel feature that limits, accounts for, and isolates the resource usage (CPU, memory, disk I/O, etc.) of a collection of processes.

Below shows how `systemd` is started, and how `systemd` is a daemon that manages other daemons.

Given a typical Linux boot process:

1. system startup: BIOS firmware, searching for the bootable device (CD-ROM, USB flash drive, a partition on a hard disk)
2. bootloader stage: load the Linux kernel image into memory
3. kernel stage: sets up interrupt handling (IRQs), mounts root filesystem, and `initramfs` (a.k.a. early user space) to detect device drivers
4. init process: `/sbin/init` (one popular implementation is `systemd`) runs as the first (user space) process such that PID = 1, and will be the last process to terminate; `systemd` prepares the user space

where in the final step during `/sbin/init`, the Linux kernel loads `systemd` and passes control over to it and the startup process begins.

<div style="display: flex; justify-content: center;">
      <img src="imgs/startup-systemd.png" width="30%" height="60%" alt="startup-systemd" />
</div>
</br>

P.S. typical PIDs are

* PID = 0: CMD: `kernel_task`, responsible for paging, and this process is always referred to as the swapper or sched process, or just cpu idle
* PID = 1: CMD: `/sbin/init`
* PID = 2: CMD: `kthreadd`

Do NOT get confused that

||`systemd`|`systemctl`|`sysctl`|
|-|-|-|-|
|**Description**|A comprehensive system and service manager for Linux|A utility to interact with `systemd`, allowing users to manage services and check their statuses.|A utility to query and modify kernel parameters at runtime|
|**Examples**||`systemctl start <service>`: Starts a service.|`sysctl -a`: Lists all available kernel parameters.|

### Service Type (Process Management)

`systemd` manages services by `systemctl start <service>` starting the service's process, and `systemctl stop <service>` killing the process.

There are different service types that cater for different purposes, e.g., restart on different booting/failure conditions and forked child process management.

Reference: https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html

* Simple

If service type is not manually set, `simple` is used.

In the default `simple` service type, only ONE process can be managed.
If multiple processed start under this service type, such as

```sh
for i in {1..5}; do
  ./start_a_process.sh &
done
```

only the 5th process survives and the previous four processes are dead.

* forking

Only the parent process exits and the forked child processes survive.

* oneshot

Parent process starts sub-processes then exit; should set `RemainAfterExit=yes` as well to prevent parent process terminating all sub-processes.
often used after OS booting for one time setup tasks.

### Resource Allocation and `cgroups`



### Register A `systemctl` Service

Reference: https://unix.stackexchange.com/questions/236084/how-do-i-create-a-service-for-a-shell-script-so-i-can-start-and-stop-it-like-a-d

1. Prepare a script/executable `/usr/bin/myscript` and make it `chmod +x`.

2. Write down below cfg to `/etc/systemd/system/my-svc.service`.

```txt
[Unit]
Description=My Script

[Service]
Type=simple
ExecStart=/usr/bin/myscript

[Install]
WantedBy=multi-user.target
```

3. Reload all systemd service files: `systemctl daemon-reload`

4. Check that it is working by starting the service with `systemctl start my-svc`.

5. To auto-start, use enable  `systemctl enable my-svc`.

#### `service` vs `systemctl`

Both are used to manage Linux processes for initialization.
`systemctl` aims to replace `service` as the modern unified process initialization management solution under `systemd`.

|`service`|`systemctl`|
|-|-|
|operates on `/etc/init.d`|operates on `/lib/systemd`|
|belongs to *SysVinit* (System V Init), aka the classic Linux initialization process|belongs to `systemd`, the successor of SysVinit and the modern initialization process|


#### One Service Managing Multiple Processes

Split processes into such that

```conf
[Unit]
Description=Simple Service Managing Multiple Processes

[Service]
Type=simple
ExecStartPre=/path/to/pre_process_1
ExecStartPre=/path/to/pre_process_2
ExecStart=/path/to/main_process
ExecStartPost=/path/to/post_process_1
ExecStartPost=/path/to/post_process_2

[Install]
WantedBy=multi-user.target
```

An alternative would be using `Type=forking` or `Type=oneshot` such that starting processes by such below `start_procs.sh`

```sh
/path/to/pre_process_1 &
/path/to/pre_process_2 &
/path/to/main_process &
/path/to/post_process_1 &
/path/to/post_process_2 &
```

Then in `service` define

```conf
[Unit]
Description=Simple Service Managing Multiple Processes

[Service]
Type=oneshot
ExecStart=bash start_procs.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

#### Debug

The registered `systemd` log can be found by `journalctl`.

## Linux Process Management

* Foreground vs Background Processes

* `nohup` and the `SIGHUP` signal

The `SIGHUP` signal to notify processes that the terminal or controlling process has been closed.
As a result, the sub-processes should exit as well.

For example, when a shell terminal is closed, the shell running processes are shutdown.

* Start From Local Shell

When started a process from a local shell, the shell becomes the parent of that process.

If the shell exits, the parent of the process is typically reassigned to the init process (PID 1).

## Common DevOps

* Port Listening

```bash
sudo lsof -i -P -n | grep LISTEN
```

* Find the largest files

By directory (One common crash is caused by too many logs generated in a directory)

```bash
sudo du -a / 2>/dev/null | sort -n -r | head -n 20
```

By file

```bash
sudo find / -type f -printf "%s\t%p\n" 2>/dev/null | sort -n | tail -10
```

* Check disk usage

```bash
df
```

* Check I/O to/from devices

```bash
iostat
```

* Failed `apt install` for connection error

There are different apt mirrors with different levels of legal constraints: *main*, *restricted*, *universe*, *multiverse*

Change apt-get mirror on `/etc/apt/sources.list` and add more mirrors to this list (below `focal` is used for ubuntu 20.04)
```bash
deb http://archive.ubuntu.com/ubuntu/ focal main universe multiverse restricted
deb http://us.archive.ubuntu.com/ubuntu/ focal main universe multiverse restricted
deb http://cn.aarchive.ubuntu.com/ubuntu focal main universe multiverse restricted
```

We can download manually from browser and instally locally:
```bash
sudo apt install ./path/to/deb
```


* Create a bootable usb

```bash
sudo umount /dev/sda1
sudo dd if=/path/to/ubuntu.iso of=/dev/sda1 bs=1M
```

* Route

`/etc/hostname` contains name of the machine, as known to applications that run locally.

`/etc/hosts` contains the mapping of some hostnames to IP addresses before DNS can be referenced. 

```
IPAddress     Hostname    		 Alias
127.0.0.1			localhost	 	 deep.openna.com
208.164.186.1		deep.openna.com		 deep
208.164.186.2		mail.openna.com		 mail
208.164.186.3		web.openna.com		 web
```

`/etc/gateways` file identifies gateways for a routed daemon.

`/etc/resolv.conf` file is used for domain name resolver

In Linux there is a *resolver* performing domain name translation.
Specifically, it translates domain names to IP addresses by querying the Domain Name Server (DNS). 
The `/etc/resolv.conf` file is the file that configures the domain name resolver.

For example, `8.8.4.4` is the Google DNS server.
```bash
nameserver 8.8.4.4
```

* `/etc` vs `/var`

`/etc` (etcetera) is used to store config, while `/var` (variable) stores frequently changed data such as logs.

* `modprobe`

The Linux kernel has a modular design. 
A kernel module, or often referred to as a driver, is a piece of code that extends the kernel’s functionality. 

Modules can be manually loaded by `modprobe`, or automatically at boot time using /etc/modules or /etc/modules-load.d/*.conf files.

To load a module
`modprobe <module_name>`

To check a module
`lsmod | grep <module_name>`

To remove a module
`modprobe -r <module_name>`
