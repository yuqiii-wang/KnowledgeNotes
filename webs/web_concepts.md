# Some Web Concepts

## MicroServices

In short, the microservice architectural style is an approach to developing a single application as a suite of small services, each running in its own process and communicating with lightweight mechanisms, often an HTTP resource API. 

## WebSocket

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