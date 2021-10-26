# Squid Proxy

Squid is a caching and forwarding HTTP web proxy, acting NIC (network interface controller/card) to control requests, and responsbile for caching frequentlly used data, etc.

Config squid on `/etc/squid/squid.conf` and run `systemctl restart squid.service` to restart squid.


```bash
# allow access from local network 10.1.1.0/16 and 10.1.2.0/16
acl our_networks src 10.1.1.0/16 10.1.2.0/16
http_access allow our_networks

# control hours of access
acl liquidweb src 10.1.10.0/24
acl liquidweb time M T W T F 9:00-17:00
```

## Squid vs Nginx

* Squid 

A "normal" proxy, such as squid http proxy, socks, etc. fetches content on end user behalf, and sits in front of end users, making TCP/IP calls out to the internet web servers and ideally caching content.

* Nginx 

When run as a reverse proxy, sits in front of server endpoints, usually load balancing between them.... therefore "reverse"... do not applicable to your needs.