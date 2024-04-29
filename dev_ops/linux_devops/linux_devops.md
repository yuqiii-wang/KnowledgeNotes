# Some Linux Knowledge and DevOps

## `service` vs `systemctl`

Both are used to manage Linux processes for initialization.

|`service`|`systemctl`|
|-|-|
|operates on `/etc/init.d`|operates on `/lib/systemd`|
|belongs to SysVinit (System V Init), aka the classic Linux initialization process|belongs to systemd, the successor of SysVinit and the modern initialization process|

### Register A `systemctl` Service

Reference: https://unix.stackexchange.com/questions/236084/how-do-i-create-a-service-for-a-shell-script-so-i-can-start-and-stop-it-like-a-d

1. Prepare a script/executable `/usr/bin/myscript` and make it `chmod +x`.

2. Write down below cfg to `/etc/systemd/system/my.service`.

```txt
[Unit]
Description=My Script

[Service]
Type=forking
ExecStart=/usr/bin/myscript

[Install]
WantedBy=multi-user.target
```

3. Reload all systemd service files: `systemctl daemon-reload`

4. Check that it is working by starting the service with `systemctl start my`.



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
