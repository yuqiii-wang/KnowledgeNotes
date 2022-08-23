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