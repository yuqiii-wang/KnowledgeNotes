# Some Linux Cmds

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