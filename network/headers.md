# Headers

## `Content-Type`

`Content-Type` specifies media type, such as
```bash
Content-Type: text/plain
Content-Type: application/json
Content-Type: application/octet-stream
Content-Type: application/x-www-form-urlencoded
```
which defines payload format; most typical are json and x-www-form-urlencoded for application data transmission.

## `Forwarded`

This header is used for debugging, statistics, and generating location-dependent content. By design, it exposes privacy sensitive information, such as the IP address of the client. 

Typical usage example: microservices where many program transfer data to each other with this header for delegation identity process.

```
Forwarded: by=<identifier>;for=<identifier>;host=<host>;proto=<http|https>
```

* `for` is Optional

    The client that initiated the request and subsequent proxies in a chain of proxies. The identifier has the same possible values as the by directive.

* `host` is Optional

    The Host request header field as received by the proxy.

* `proto` is Optional

    Indicates which protocol was used to make the request (typically "http" or "https").

For example,
```
Forwarded proto=https;host=example.server.com;for=example.client.com
```

## `Accept`

The Accept request HTTP header indicates which content types, expressed as MIME types, the client is able to understand, such as `Accept: text/html, application/xhtml+xml`

To accept all, `Accept: */*`