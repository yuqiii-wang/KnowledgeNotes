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