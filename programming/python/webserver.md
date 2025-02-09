# Webserver

There are three popular webservers with python as backend.

* Flask
* FastAPI
* Django

||Flask|FastAPI|Django|
|-|-|-|-|
|Primary Standard|WSGI|ASGI|WSGI (default) + ASGI (since v3.0)|
|Built On|Werkzeug (WSGI toolkit)|`Starlette` (ASGI framework) & `Pydantic` (data validation)|Custom HTTP handling (WSGI), with new ASGI layer|
|Supported servers|Werkzeug (dev), Gunicorn, uWSGI (prod)|Uvicorn, Hypercorn|Gunicorn/uWSGI (WSGI) or Daphne/Uvicorn (ASGI)|

By Jan 2025, FastAPI is the optimal for most use cases.

## Protocols

* WSGI (Web Server Gateway Interface)

Definition: A synchronous Python standard (PEP 3333) for communication between web servers and Python apps. Designed for blocking I/O.

* ASGI (Asynchronous Server Gateway Interface)

Definition: A synchronous + asynchronous Python standard (ASGI spec) that extends WSGI. Supports async/await for non-blocking I/O.

## Servers

|Server|Protocol|Key Features|Description|Used With|Launched Year|
|-|-|-|-|-|-|
|Werkzeug|WSGI|Simple as a WSGI utility tool|A WSGI utility library used by Flask for routing, request/response handling|Flask|~2007|
|Gunicorn|WSGI/ASGI|Production-grade, process management|Uses a pre-fork worker model (each worker is a separate process) for handling requests.|Flask, Django, FastAPI|~2009|
|Daphne|ASGI|WebSocket support, Django integration|An ASGI server originally built for Django Channels (WebSocket support)|Django (ASGI mode)|~2015|
|Uvicorn|ASGI|Async, lightweight, FastAPI's default|A lightweight ASGI server built on `uvloop` (a fast, drop-in replacement for Python's default event loop written in C) and `httptools` (a fast HTTP parser). Optimized for FastAPI and Starlette.|FastAPI, Starlette|~2018|
|Hypercorn|ASGI|HTTP/2, WebSockets, Gunicorn-like|ASGI server inspired by Gunicorn, supporting HTTP/2 and WebSockets.|FastAPI, Quart|~2018|

## Core Libraries

||Framework|Purpose|Key Feature|
|-|-|-|-|
|Starlette|FastAPI|Async routing, middleware|ASGI foundation|
|Werkzeug|Flask|WSGI routing, request handling|Flask's backbone|
|Pydantic|FastAPI|Data validation, serialization|Type hints + OpenAPI|
|Jinja2|Flask|HTML templating|Server-side rendering|