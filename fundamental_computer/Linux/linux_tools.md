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

* `awk` provides a scripting language for text processing, of a given syntax `aws {cmd_scripts} filename`. Var denoted by `$` indicates str separated by `empty space`; `$0` represents the whole input str.

```bash
echo "Hello Tom" | awk '{$2="Adam"; print $0}'

# put awk_cmd in a file and run it
awk -f mypgoram.awk input.txt
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

## Concepts

### Process

* PID - Process ID

Process ID is a unique identifier of a process.

* PPID - Parent Process ID

The parent process ID of a process is the process ID of its creator, for the lifetime of the creator. After the creator's lifetime has ended, the parent process ID is the process ID of an implementation-defined system process.

* SID - Session ID

A collection of process groups established for job control purposes. Each process group is a member of a session.

* PGID - Process Group ID

A collection of processes that permits the signaling of related processes.

* EUID - Effective User ID

An attribute of a process that is used in determining various permissions, including file access permissions; see also User ID.

### Service vs Systemctl

`Service` is an "high-level" command used for start, restart, stop and status services in different Unixes and Linuxes, operating on the files in `/etc/init.d`.

`systemctl` operates on the files in `/lib/systemd`.

`service` is a **wrapper** for all three init systems (/init, systemd and upstart).

* Mask/Unmask a service

We should mask a service, if we want to prevent any kind of activation, even manual. e.g. If we don’t want to apply firewall rules at all then we can mask the `firewalld` service.

```bash
systemctl unmask firewalld
systemctl start firewalld
```

### Kernel Levels:

A `runlevel` can simply be thought of as the state your system enters like if a system is in a single-user mode it will have a runlevel 1 while if the system is in a multi-user mode it will have a runlevel 5.

LINUX kernel supports these seven different runlevels :

0 – System halt i.e the system can be safely powered off with no activity.

1 – Single user mode.

2 – Multiple user mode with no NFS(network file system).

3 – Multiple user mode under the command line interface and not under the graphical user interface.

4 – User-definable.

5 – Multiple user mode under GUI (graphical user interface) and this is the standard runlevel for most of the LINUX based systems.

6 – Reboot which is used to restart the system.