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

## Docker

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

## JUnit Test
A JUnit test is a method contained in a class which is only used for testing. This is called a Test class. To define that a certain method is a test method, annotate it with the `@Test` annotation.

The following code shows a JUnit test using the JUnit 5 version. This test assumes that the MyClass class exists and has a multiply(int, int) method.

```java
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

public class MyTests {

    @Test
    public void multiplicationOfZeroIntegersShouldReturnZero() {
        MyClass tester = new MyClass(); // MyClass is tested

        // assert statements
        assertEquals(0, tester.multiply(10, 0), "10 x 0 must be 0");
        assertEquals(0, tester.multiply(0, 10), "0 x 10 must be 0");
        assertEquals(0, tester.multiply(0, 0), "0 x 0 must be 0");
    }
}
``` 