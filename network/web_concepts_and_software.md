# Some Web Concepts

## MicroServices

In short, the microservice architectural style is an approach to developing a single application as a suite of small services, each running in its own process and communicating with lightweight mechanisms, often an HTTP resource API. 

## WebSocket

**WebSocket** is a computer communications protocol, providing full-duplex communication channels over a single TCP connection, facilitating real-time data transfer from and to the server.

Below is an example request and response
```yaml
GET /chat HTTP/1.1
Host: server.example.com
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: x3JJHMbDL1EzLkh9GBhXDw==
Sec-WebSocket-Protocol: chat, superchat
Sec-WebSocket-Version: 13
Origin: http://example.com
```
And the server's response is 
```yaml
HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: HSmrc0sMlYUkAGmm5OPpG2HaGWk=
Sec-WebSocket-Protocol: chat
```


## Nginx

Originally designed to solve C10k problem (the problem of optimising network sockets to handle a large number of clients at the same time).

The below is an example of a server configuration.
```conf
server {
    listen         80 default_server;
    listen         [::]:80 default_server;
    server_name    example.com www.example.com;
    root           /var/www/example.com;
    index          index.html;

    gzip             on;
    gzip_comp_level  3;
    gzip_types       text/plain text/css application/javascript image/*;
}
```

## Wiremock

**Wiremock** is a simulator for HTTP-based APIs.

For example, you want to build a ticket selling system, but have only finished the front/back-end and want to test with no real data (e.g., PNR (Passenger Name Record) from external sources).

A quick tutorial:
```bash
java -jar wiremock-standalone-x.xx.x.jar # x.xx.x is the version number

# setup mock response
curl -X POST --data '{"request": {"url":"index", "method":"GET"}, "response": {"status":200, "body":"Hello World"}}' http://localhost:8080/__admin/mappings/new

# Go to your browser to  http://localhost:8080/index
```

## Headless Browser
A headless browser is a web browser without a graphical user interface.

## Web Agents vs J2EE Agents

Both act as an agent for Policy Decision Process, performing tasks ranging from user/client authentication to ip or location validation.

* **Web agents** control access to content on web servers and proxy servers. Most typical is a web browser, that stores cookies, for instance, to hold user credentials for SSO before accessing a server's resources.

* **J2EE agent** can be installed for protecting a variety of hosted J2EE applications, which might require a varying set of security policy implementation. The agent is a servlet filter that intercepts inbound requests to a server, including authorization/validation of the requests by defined policies. Policies can be declarative (e.g., xml) or programatic (e.g., java code).

## Web Server

A web server is computer software and underlying hardware that accepts requests via HTTP/HTTPS to distribute web content.

Key services include:
* URL path translation and resource mapping
* Bandwidth throttling
* http/https connection management
* Logging
* Request authentication
