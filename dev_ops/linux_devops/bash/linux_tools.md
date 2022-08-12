# Some Linux Knowledge

## Useful Tools

* `pushd` and `popd`: The end result of pushd is the same as cd, with a new origin set.

```bash
$ pwd
one
$ pushd two/three
~/one/two/three ~/one
$ pwd
three
```

* `chkconfig` command is used to list all available services and view or update their run level settings.
However, this command is **NO** longer available in Ubuntu.The equivalent command to chkconfig is `update-rc.d`. They can be used to `add` services upon according to runlevel the computer is at.

```bash
chkconfig --add name
chkconfig [--level levels] name 
```

* `tee` command reads the standard input and writes it to both the standard output and one or more files. 

```bash
cat test_text >> file1.txt
wc -l file1.txt | tee -a file2.txt
```

* `tr` is used to transform string or delete characters from the string. Other options include `-d` for delete.

```bash
# transform a string from lowercase to uppercase
echo linuxhint | tr a-z A-Z
```

* `chage` command is used to view and change the user password expiry information.

```bash
# Show account aging information.
sudo chage -l root

# Set the minimum number of days between password changes to MIN_DAYS. A value of zero for this field indicates that the user may change his/her password at any time.
sudo chage -m 0 username

# Set the maximum number of days during which a password is valid. 
sudo chage -M 99999 username
```

* `systemd` is a software suite that provides an array of system components for Linux operating systems, of which primary component is a "system and service manager"—an init system used to bootstrap user space and manage user processes. `systemctl` is the central management tool for controlling the init system.

P.S. The name systemd adheres to the Unix convention of naming daemons by appending the letter d.

* In Unix-based computer operating systems, `init` (short for initialization) is the first process started during booting of the computer system. Init is a daemon process that continues running until the system is shut down. 

```bash
sudo systemctl start application.service
```

* `sed` is a stream editor. A stream editor is used to perform basic text transformations on an input stream (a file or input from a pipeline).

```bash
# To replace all occurrences of ‘hello’ to ‘world’ in the file input.txt:
# Below are equivalent 
echo "Hello World, hello world Ohhh Yeah" >> input.txt
sed 's/hello/world/' input.txt > output.txt
sed 's/hello/world/' < input.txt > output.txt
cat input.txt | sed 's/hello/world/' - > output.txt
rm input.txt && rm output.txt
```

* sshd

`sshd` is the OpenSSH server process. It listens to incoming connections using the SSH protocol and acts as the server for the protocol. It handles user authentication, encryption, terminal connections, file transfers, and tunneling.

The `sshd` process is started when the system boots. The program is usually located at `/usr/sbin/sshd`. It runs as `root`.

* `keytool` is used to manage keys and certificates and store them in a keystore. The `keytool` command allows us to create self-signed certificates and show information about the keystore.

* `chkconfig` command is used to list all available services and view or update their run level settings. In simple words it is used to list current startup information of services or any particular service

* ln

A soft link is something like a shortcut in Windows. It is an indirect pointer to a file or directory. Unlike a hard link, a symbolic link can point to a file or a directory on a different filesystem or partition.

```bash
ln -s [OPTIONS] FILE LINK
```

* read os name

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
echo $distribution
```
