# SSH (Secure Shell)

SSH provides secure communication between computers.

SSH usually runs over TCP.

It sought to address the security lapses of Telnet, a protocol that allows one computer to log into another on the same open network.

## Quick Setup

1. Generate key pair by ssh (choose no passphrase) on local computer.

```bash
ssh-keygen -t rsa
```

The above cmd should produce two files.

```bash
# on local computer
$HOME/.ssh/id_rsa # Your private RSA key
$HOME/.ssh/id_rsa.pub # Your public RSA key
```

2. Share the `id_rsa.pub` to the remote server.

The remote server should have `ssh` already installed.
Usually, `id_rsa.pub` is copied to the below file.

```bash
# on remote server
$HOME/.ssh/authorized_keys
```

Use the same `id_rsa`/`id_rsa.pub` pair for multiple remote servers.
This means sharing the same `id_rsa.pub` to multiple servers rather than repeatedly generating key pairs by `ssh-keygen`.

3. Find if the remote server host has already existed in local computer.
If so, delete it.

```bash
# on local computer
cat /Users/$HOME/.ssh/known_hosts | grep '<remote_host>'
```

4. The remote server should have its `id_rsa.pub` copied to local computer to `authorized_keys`.

```bash
# on local computer
$HOME/.ssh/authorized_keys
```

This step might be auto completed for some open-to-public servers such as `github.com`.

Now, ssh should work as per above setup.

### `authorized_keys` vs `known_hosts`

* `authorized_keys`

Holds a list of authorized public keys for servers. When the client connects to a server, the server authenticates the client by checking its signed public key stored within this file

* `known_hosts`

The first time a client connects to a server, the client needs to check if the public key presented by the server is really the public key of the server (this is why the first time connection requiring user verifying if a public key is trusted).
If the server's public key has already existed in client's `known_hosts`, client can just `ssh` to the server.

### `scp` VS `sftp`

`scp` uses `ssh` to file transfer `scp <fromDirectory> <toDirectory>`

For example, to download from a remote computer, run

```bash
scp -r yuqi@<remote_server>:/path/to/remote/server /path/to/local/server
```

`sftp` is more elaborate than `scp`, and allows interactive commands similar to `ftp`.

For example, use `ftp put` to copy multiple files to remote server.

```bash
sftp username@${remotehost} << EOF
   put $localpath/fileA $remotepath/
   put $localpath/fileB $remotepath/
EOF
```

### SSH vs Telnet

||Telnet|SSH|
|-|-|-|
|Name|Telecommunications and Networks (Telnet)|Secure Shell (SSH)|
|Data Format|simple plain text|encrypted text|
|Data Format|simple plain text|encrypted text|
|Authentication|No authentication|key pair + username/password|

## The Underlying Mechanism

SSH protocol starts building from *transport layer*, typically over TCP.
The SSH protocol is composed of the below three protocols over TCP.

* The Secure Shell (SSH) Transport Layer Protocol (rfc4253)

This protocol runs over TCP, and it can be used as a basis for a number of secure network services.
It provides strong encryption, server authentication, and integrity protection.

Key exchange method, public key algorithm, symmetric encryption algorithm, message authentication algorithm, and hash algorithm are all negotiated.

* The Secure Shell (SSH) Authentication Protocol (rfc4252)

This protocol runs over rfc4253, and it provides a single authenticated tunnel for the SSH connection protocol.
The authentication methods include public key, password, etc.

* The Secure Shell (SSH) Connection Protocol (rfc4254)

This protocol runs over rfc4253 and rfc4252, and it provides interactive login sessions, remote execution of commands, forwarded TCP/IP connections, and forwarded X11 connections, that are multiplexed into one tunnel.

### SSH Packet

One SSH packet is shown as below: length + payload (up to 32,768 bytes each) + MAC addr

<div style="display: flex; justify-content: center;">
      <img src="imgs/ssh_packet.png" width="60%" height="30%" alt="ssh_packet" />
</div>
</br>