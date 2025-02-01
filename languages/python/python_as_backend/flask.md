# Flask

Flask is a lightweight web framework written in Python.

## High-Throughput and Session Management

Use nginx as a load balancer to distribute requests to multiple flask instances.

## Core Component: Werkzeug WSGI

Flask relies on Werkzeug, a WSGI (Web Server Gateway Interface) toolkit to handle HTTP requests, including underlying operations e.g., TCP/IP routing, multiplexing I/O over socket.

There are two popular WSGI implementations:

* Gunicorn
* uWSGI