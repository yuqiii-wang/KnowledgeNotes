# Network Knowledge

## A typical full HTTPS flow

### CDN

### HTTP Cache in Frontend

## DNS (Domain Name System)

DNS server maps between IPs and hostnames so such as "www.example.com" to an IP addr.

For example, `8.8.8.8` is a Google provided DNS server that is publicly available for searching IP/hostname mapping globally.

## DHCP (Dynamic Host Configuration Protocol)

DHCP server assigns IP to device.

For example, if a computer is connected to a hotspot of a phone for data roaming, the phone acts as a DHCP server assigning an IP to the computer.

by `ipconfig getpacket en0`, there is below, where

* `yiaddr` = local computer IP address
* `siaddr` = DHCP Server Address
* `giaddr` = gateway IP address
* `chaddr` = Hardware Address

```txt
op = BOOTREPLY
htype = 1
flags = 0
hlen = 6
hops = 0
xid = 0x000000
secs = 0
ciaddr = 0.0.0.0
yiaddr = 172.20.10.x
siaddr = 172.20.10.x
giaddr = 0.0.0.0
chaddr = xx:xx:1:xx:xx:xx
sname = Yuqis-iPhone
file =
options:
Options count is 7
dhcp_message_type (uint8): ACK 0x5
server_identifier (ip): 172.20.10.x
lease_time (uint32): 0x15180
subnet_mask (ip): 255.255.255.240
router (ip_mult): {172.20.10.x}
domain_name_server (ip_mult): {172.20.10.x}
end (none):
```

## Network Time Protocol

Allow computers to access network unified time to get time synced. 

Stratum Levels:

* Stratum 0: Atomic clocks
* Stratum 1 - 5: various Time Servers
* Stratum 16: unsynced

Time authentication fails for large time gaps.

Primary NTP servers provide first source time data to secondary servers and forward to other NTP servers, and NTP clients request for time sync info from these NTP servers.
