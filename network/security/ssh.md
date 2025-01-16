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

## Under `$HOME/.ssh`

The `.ssh` directory in a user's home folder contains several important files used for SSH authentication and connection management.
They are `authorized_keys`, `id_rsa`, `id_rsa.pub`, and `known_hosts`.

When user computer connects to a remote computer, the user computer is named *client*, and the remote computer is *server*.

### `authorized_keys` vs `known_hosts`

In summary,

* Server checks client's public key against `authorized_keys`.
* Client verifies server's public key against `known_hosts`.

### File Explanation

#### `authorized_keys`

When a client attempts to connect to a server using SSH key authentication, the server checks if the client's public key matches any key in the authorized_keys file.

Each line contains a single public key:
`ssh-rsa AAAAB3NzaC1yc2EAAA... user@host`

#### `known_hosts`

When a client connects to a server for the first time, the server's public key is added to client's `known_hosts`.
If the server's public key has already existed in client's `known_hosts`, client can just `ssh` to the server.

Each line contains a server's hostname, IP address, and public key:
`server.example.com ssh-rsa AAAAB3NzaC1yc2EAAA...`

#### `id_rsa`

The private key is used to prove the client's identity to the server during SSH key authentication.

#### `id_rsa.pub`

The public key corresponding to the private key (`id_rsa`).

The public key is added to the `authorized_keys` file on the server to allow the client to authenticate.

Example content of `id_rsa.pub`:
`ssh-rsa AAAAB3NzaC1yc2EAAAABIwAAAQEArV1... user@host`

### How Server Authenticates Client

1. Client sends public key

When the client attempts to connect to the server, it sends its public key (`id_rsa.pub`) to the server.

2. Server checks `authorized_keys`, and init *challenge-response process*

The server looks for the client's public key in the `~/.ssh/authorized_keys` file.

If existed, the server proceeds with the challenge-response process.

The challenge-response process begins with server generating a random byte string (challenge text) and send it to client.

3. Client "solves" the challenge and responds with the result

Client uses its private key `id_rsa` to sign/encrypt the challenge text.

4. Server verifies the client's response for authentication

Server decrypts the response with stored public key in `authorized_keys`, and check if the decrypted text matches the original challenge text

#### Challenge-Response Process

1. On server: random text challenge generation

```sh
openssl rand -base64 32 > challenge.txt
```

2. On client: sign the challenge text

```sh
openssl pkeyutl -sign -inkey id_rsa -in challenge.txt -out challenge.sig
```

3. On server: verify the signature

```sh
openssl pkeyutl -verify -pubin -inkey authorized_keys -in challenge.txt -sigfile challenge.sig
```

## Related Protocols

### `scp` VS `sftp`

* `scp` uses `ssh` to file transfer `scp <fromDirectory> <toDirectory>`

For example, to download from a remote computer, run

```bash
scp -r yuqi@<remote_server>:/path/to/remote/server /path/to/local/server
```

* `sftp` also built on top of `ssh`, but is more elaborate than `scp`, and allows interactive commands similar to `ftp`.

For example, use `ftp put` to copy multiple files to remote server.

```bash
sftp username@${remotehost} << EOF
   put $localpath/fileA $remotepath/
   put $localpath/fileB $remotepath/
EOF
```

Remember, `sftp` only tries to behave (many similar commands/options) like `ftp`, but build on `ssh`.

|FTP|SFTP|
|-|-|
|Port 21 for control; Port 20 for data.|Port 22 (same as SSH).|
|No encryption|Encrypted using SSH|

### SSH vs Telnet

||Telnet|SSH|
|-|-|-|
|Name|Telecommunications and Networks (Telnet)|Secure Shell (SSH)|
|Data Format|simple plain text|encrypted text|
|Authentication|No authentication|key pair + username/password|
|port|23 (default)|22 (default)|

## Protocol Implementation

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