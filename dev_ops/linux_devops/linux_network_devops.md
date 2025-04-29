# Linux Network DevOps

## Common Connection Test Tools (Host Reachable)

* `ping`

`ping` sends ICMP (Internet Control Message Protocol) echo request packets to a target device and waits for an echo reply.

* `telnet`

`telnet` establishes a TCP connection to a specified port and allows text-based communication.

* `ssh`

`ssh` establishes an encrypted connection to a remote server using the SSH protocol (default port 22).

* `nslookup`

`nslookup` checks Domain Name bounded IP address.

For example, `nslookup github.com` gives DNS server and queried server IP.

```txt
Server:         8.8.8.8
Address:        8.8.8.8#53

Non-authoritative answer:
Name:   github.com
Address: 20.205.243.166
```

where

-> `Server:         8.8.8.8` is the DNS server
-> `#53` indicates that the request is being sent to port 53, which is the default port for DNS communication using UDP (by default) or TCP.
-> `Non-authoritative answer`: This means that the DNS server (20.205.243.166) is returning cached data rather than retrieving it directly from the authoritative DNS server.

* `tracert`/`traceroute`

Run traceroute using the following commands:

Windows: `tracert example.com`

Linux/Mac: `traceroute example.com`

Traceroute increments the TTL (Time To Live) by 1 for each subsequent packet, revealing the next hop in the path.

```txt
1  192.168.1.1  1 ms  1 ms  1 ms  (Your router)
2  10.0.0.1     5 ms  6 ms  5 ms  (ISP gateway)
3  203.0.113.45 10 ms 11 ms 10 ms (ISP backbone)
4  198.51.100.2 20 ms 21 ms 20 ms (Intermediate router)
5  93.184.216.34 30 ms 31 ms 30 ms (Destination server)
```

If `traceroute` shows many `*`s, it indicates that server does not respond.

Some server might disable UDP/TCP, one can switch protocol by

```sh
# to use ICMP
traceroute -I example.com

# to use TCP
traceroute -T -p 80 example.com
```

### Comparisons Between The Tools

||Ping (ICMP)|SSH (Secure Shell)|Telnet|nslookup|Traceroute|
|-|-|-|-|-|-|
|Protocol|ICMP|SSH (TCP-based)|Telnet (TCP-based)|DNS protocol (RFC 1035)|Windows `tracert`: ICMP (default), Unix/Linux `traceroute`: UDP (default), TCP, ICMP|
|Layer|Network (Layer 3)|Application (Layer 7)|Application (Layer 7)|Application (Layer 7):DNS, Transport (Layer 4): UDP (default)/TCP|Network/Transport (Layer 3/4)|
|Port|None|22 (default)|23 (default)|53 (default)|None|

## HTTPS Check

### Full HTTPS Handshake Check with `curl`

With sent `curl -v https://example.com`, there are three protocols for communication.

```txt
[TCP Handshake]
Client: SYN → Server
Server: SYN-ACK → Client
Client: ACK → Server

[TLS Handshake]
Client: ClientHello (TLS 1.3, SNI=example.com) → Server
Server: ServerHello (TLS 1.3), Certificate → Client
Client: Key Exchange, Finished → Server
Server: Finished → Client

[HTTP Request]
Client: GET / HTTP/1.1 → Server (encrypted)
Server: HTTP/1.1 200 OK → Client (encrypted)
```

#### TCP Handshake Success/Failure Check

Successful Connection by `curl -v` log:

```txt
* Trying 93.184.216.34:443...
* Connected to example.com (93.184.216.34) port 443 (#0)
```

Failed Connection by `curl -v` log:

```txt
* Trying 93.184.216.34:443...
* connect to 93.184.216.34 port 443 failed: Connection timed out
```

#### TLS Handshake Success/Failure Check

Successful Handshake by `curl -v` log:

```txt
* TLSv1.3 (OUT), TLS handshake, Client hello (1):
* TLSv1.3 (IN), TLS handshake, Server hello (2):
* TLSv1.3 (IN), TLS handshake, Encrypted Extensions (8):
* TLSv1.3 (IN), TLS handshake, Certificate (11):
* TLSv1.3 (IN), TLS handshake, CERT verify (15):
* TLSv1.3 (IN), TLS handshake, Finished (20):
* TLSv1.3 (OUT), TLS handshake, Finished (20):
* SSL connection using TLSv1.3 / AES256-GCM-SHA384
```

Failed Handshake by `curl -v` log:

```txt
* SSL certificate problem: unable to get local issuer certificate
```

#### HTTP Handshake Success/Failure Check

Successful Handshake by `curl -v` log:

```txt
> GET / HTTP/1.1
> Host: example.com
> User-Agent: curl/7.79.1
> Accept: */*
>
< HTTP/1.1 200 OK
< Content-Type: text/html
< Content-Length: 1234
```

Failed Handshake by `curl -v` log:

```txt
> GET / HTTP/1.1
> Host: example.com
> User-Agent: curl/7.79.1
> Accept: */*
>
* Empty reply from server
* Connection #0 to host example.com left intact
```

### Cert/TLS/SSL Check by `openssl s_client`

`openssl s_client` reveals server certificate.

For example, `openssl s_client Baidu.com:443` shows a certificate chain.

```txt
CONNECTED(00000006)
depth=2 C = US, O = DigiCert Inc, OU = www.digicert.com, CN = DigiCert Global Root G2
verify return:1
depth=1 C = US, O = "DigiCert, Inc.", CN = DigiCert Secure Site Pro G2 TLS CN RSA4096 SHA256 2022 CA1
verify return:1
depth=0 C = CN, ST = \E5\8C\97\E4\BA\AC\E5\B8\82, O = "BeiJing Baidu Netcom Science Technology Co., Ltd", CN = www.baidu.cn
verify return:1
---
Certificate chain
 0 s:C = CN, ST = \E5\8C\97\E4\BA\AC\E5\B8\82, O = "BeiJing Baidu Netcom Science Technology Co., Ltd", CN = www.baidu.cn
   i:C = US, O = "DigiCert, Inc.", CN = DigiCert Secure Site Pro G2 TLS CN RSA4096 SHA256 2022 CA1
   a:PKEY: rsaEncryption, 2048 (bit); sigalg: RSA-SHA256
   v:NotBefore: Feb 12 00:00:00 2025 GMT; NotAfter: Mar  3 23:59:59 2026 GMT
 1 s:C = US, O = "DigiCert, Inc.", CN = DigiCert Secure Site Pro G2 TLS CN RSA4096 SHA256 2022 CA1
   i:C = US, O = DigiCert Inc, OU = www.digicert.com, CN = DigiCert Global Root G2
   a:PKEY: rsaEncryption, 4096 (bit); sigalg: RSA-SHA256
   v:NotBefore: Dec 15 00:00:00 2022 GMT; NotAfter: Dec 14 23:59:59 2032 GMT
 2 s:C = US, O = DigiCert Inc, OU = www.digicert.com, CN = DigiCert Global Root G2
   i:C = US, O = DigiCert Inc, OU = www.digicert.com, CN = DigiCert Global Root CA
   a:PKEY: rsaEncryption, 2048 (bit); sigalg: RSA-SHA256
   v:NotBefore: Jan 18 00:00:00 2024 GMT; NotAfter: Nov  9 23:59:59 2031 GMT
---
Server certificate
-----BEGIN CERTIFICATE-----
...
-----END CERTIFICATE-----
...
```

## Network Check

### Show Network Traffic

* Port Listening

```bash
sudo lsof -i -P -n | grep LISTEN
```


### Debug: `ping` works but `ssh` failed

If `ping` can reach the destination server, it means the connection is ok, and the remote server is likely up and running.

#### `ssh` got timeout

Check if the port is open

```sh
telnet remote_server 22
```

#### `ssh` responds with auth error

Errors like below show authentication error:

* Permission denied (publickey)
* No supported authentication methods available
* Agent admitted failure to sign using the key

```sh
ssh -vvv user@remote_server
```

## DNS Resolution

DNS (Domain Name System) is a system that translates human-readable domain names (e.g., `example.com`) into IP addresses (e.g., `93.184.216.34`) that computers can understand.

### Config Files

* `/etc/hostname` contains name of the machine, as known to applications that run locally.

* `/etc/hosts` contains the mapping of some hostnames to IP addresses before DNS can be referenced.

```txt
IPAddress           Hostname            Alias
127.0.0.1           localhost           deep.openna.com
208.164.186.1       deep.openna.com     deep
208.164.186.2       mail.openna.com     mail
208.164.186.3       web.openna.com      web
```

* `/etc/gateways` file identifies gateways for a routed daemon.

* `/etc/resolv.conf` file is used for domain name resolver

In Linux there is a *resolver* performing domain name translation.
Specifically, it translates domain names to IP addresses by querying the Domain Name Server (DNS).
The `/etc/resolv.conf` file is the file that configures the domain name resolver.

For example, `8.8.8.8` is the Google DNS server.

```bash
nameserver 8.8.8.8
```

#### Resolver Libraries

* `glibc` (GNU C Library): Provides the getaddrinfo() and gethostbyname() functions for DNS resolution.
* `nsswitch` (Name Service Switch): Determines the order and sources for resolving hostnames (e.g., DNS, /etc/hosts).

### DNS Resolution Process

#### Step 1: Application Requests Resolution

The application calls a resolver function like `getaddrinfo()` or `gethostbyname()` to resolve a hostname.

#### Step 2: Check Local Configuration

The resolver first checks the `/etc/hosts` file for a static mapping of the hostname to an IP address.

#### Step 3: Query DNS Servers

If the hostname is not found in `/etc/hosts`, the resolver queries DNS servers configured in /`etc/resolv.conf`.

#### Step 4: Recursive DNS Resolution

The DNS server (e.g., `8.8.8.8`) performs recursive resolution if it doesn't have the answer cached:

* Root DNS Servers: The resolver queries a root DNS server to find the authoritative server for the top-level domain (e.g., `.com`).
* TLD DNS Servers: The resolver queries the TLD DNS server to find the authoritative server for the domain (e.g., `example.com`).
* Authoritative DNS Servers: The resolver queries the authoritative DNS server for the domain to get the IP address.

#### Step 5: Return the Result

* The DNS server returns the IP address to the resolver.
* The resolver caches the result (if caching is enabled) and returns it to the application.

### Tools to Debug DNS Resolution

#### `dig` (Domain Information Groper)

Output includes the IP address, authoritative servers, and query time.

```txt
; <<>> DiG 9.10.6 <<>> github.com
;; global options: +cmd
;; Got answer:
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 33345
;; flags: qr rd ra; QUERY: 1, ANSWER: 1, AUTHORITY: 0, ADDITIONAL: 1

;; OPT PSEUDOSECTION:
; EDNS: version: 0, flags:; udp: 512
;; QUESTION SECTION:
;github.com.                    IN      A

;; ANSWER SECTION:
github.com.             60      IN      A       20.205.243.166

;; Query time: 16 msec
;; SERVER: 8.8.8.8#53(8.8.8.8)
;; WHEN: Wed Jan 15 23:09:31 CST 2025
;; MSG SIZE  rcvd: 55
```

#### `nslookup`


The underlying format of an nslookup request follows the DNS message format, which is structured according to the DNS protocol (RFC 1035).
The request is typically a binary-encoded packet sent over UDP (default) or TCP (if needed) on port 53.

A typical `nslookup example.com 8.8.8.8` request is

```txt
AA AA      → Transaction ID (randomly generated)
01 00      → Standard query (flags: recursion desired)
00 01      → 1 question
00 00      → 0 answer records
00 00      → 0 authority records
00 00      → 0 additional records
07 65 78 61 6D 70 6C 65 → "example" (length-prefixed domain name encoding)
03 63 6F 6D → "com"
00        → End of domain name
00 01     → Query Type (A record)
00 01     → Query Class (IN for Internet)
```

##### `nslookup` Example

For example, `nslookup github.com` gives DNS server and queried server IP.

```txt
Server:         8.8.8.8
Address:        8.8.8.8#53

Non-authoritative answer:
Name:   github.com
Address: 20.205.243.166
```

where

* `Server:         8.8.8.8` is the DNS server
* `#53` indicates that the request is being sent to port 53, which is the default port for DNS communication using UDP or TCP.
* `Non-authoritative answer`: This means that the DNS server (133.11.162.44) is returning cached data rather than retrieving it directly from the authoritative DNS server.

### DNS Pollution

Internet Service Provider (ISP) or malware can direct user to a wrong website (for advertisements or other purposes) via DNS pollution that DNS (usually based on UDP, easy to monitor) was intercepted and returned a false IP.

#### DNS Pollution Example

The below lookup

```sh
nslookup www.google.com 8.8.8.8
```

returns

```txt
Server:         8.8.8.8
Address:        8.8.8.8#53

Non-authoritative answer:
Name:   www.google.com
Address: 31.13.94.37
```

However, `31.13.94.37` is the IP of `instagram-p15-shv-01-eze1.fbcdn.net` (tested in Apr 2025) not Google.
This proves that the DNS is polluted.

#### DNS Pollution Mitigation with Enhanced Cyber security Protocol

HTTPS is a very commonly used protocol that is unlikely blocked.
DNS can run on HTTPS.

```sh
curl -H 'accept: application/dns-json' 'https://cloudflare-dns.com/dns-query?name=www.google.com&type=A'
```

that returns

```json
{"Status":0,"TC":false,"RD":true,"RA":true,"AD":false,"CD":false,"Question":[{"name":"www.google.com","type":1}],
    "Answer":[{
        "name":"www.google.com",
        "type":1,
        "TTL":210,
        "data":"142.251.32.36"}
    ]
}
```

## DHCP

DHCP (Dynamic Host Configuration Protocol) can provide various network services but not limited to:

* DHCP dynamically **assigns unique IP addresses** to devices/clients on the network.
* DHCP provides the **subnet mask**, which defines the network's size and structure.
* DHCP assigns the **default gateway**, e.g., `192.168.1.1` when client connects to the internet.
* DHCP provides the **IP addresses of DNS servers**, write `8.8.8.8` (Google's public DNS server) to `/etc/resolv.conf`
* DHCP assigns an IP address **lease time**, which specifies how long the client can use the assigned IP address before it must be renewed.

### DHCP Process

#### When A Client Inits DHCP Discover

* Device/computer first connection to a network, e.g., just joined a Wi-Fi network.
* DHCP Lease Expiration
* Network Change, e.g., cell phone leaves home Wi-Fi and joins a coffee shop's Wi-Fi.
* Others: Manual DHCP Renewal, IP Address Conflict, etc.

#### Locate A DHCP Server

* Local Network (Home or Office): Home routers (e.g., Netgear, TP-Link) often have a built-in DHCP server.
* ISP (Internet Service Provider)
* Virtualized Environments: In cloud or virtualized environments, DHCP services may be provided by the hypervisor or cloud provider, e.g., AWS DHCP options sets.

#### DHCP DORA Flow

The DHCP process involves four main steps, often referred to as DORA (Discover, Offer, Request, Acknowledgment):

##### Step 1: DHCP Discover

The client sends a DHCP Discover message as a broadcast to locate a DHCP server.

##### Step 2: DHCP Offer

Example OFFER response is such as below ("Server IP Address" is the DHCP server)

```txt
OpCode: 2 (Boot Reply)
Hardware Type: 1 (Ethernet)
Hardware Address Length: 6
Hops: 0
Transaction ID: 0x12345678
Seconds Elapsed: 0
Flags: 0x8000 (Broadcast)
Client IP Address: 0.0.0.0
Your IP Address: 192.168.1.100
Server IP Address: 192.168.1.1
Gateway IP Address: 192.168.1.1
Client MAC Address: 00:11:22:33:44:55
Options:
  - DHCP Message Type: DHCP Offer (2)
  - Subnet Mask: 255.255.255.0
  - Router (Default Gateway): 192.168.1.1
  - DNS Server: 8.8.8.8, 8.8.4.4
  - IP Address Lease Time: 86400 seconds (1 day)
  - Server Identifier: 192.168.1.1
```

##### Step 3: DHCP Request

The client sends a DHCP Request message to accept the offered IP address and configuration.

##### Step 4: DHCP Acknowledgment (ACK)

The DHCP server sends a DHCP Acknowledgment (ACK) message to confirm the IP address assignment and provide the network configuration.
