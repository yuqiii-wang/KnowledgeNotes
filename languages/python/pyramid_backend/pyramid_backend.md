# Pyramid as Backend Server

Install

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyramid waitress
```

Quick Start: write `app.py` with the below, then run by `python app.py`.

```python
from waitress import serve
from pyramid.config import Configurator
from pyramid.response import Response


def hello_world(request):
    print('Incoming request')
    return Response('<body><h1>Hello World!</h1></body>')


if __name__ == '__main__':
    with Configurator() as config:
        config.add_route('hello', '/')
        config.add_view(hello_world, route_name='hello')
        app = config.make_wsgi_app()
    serve(app, host='0.0.0.0', port=6543)
```

## Python As Backend Server Basics

* Web Server Gateway Interface (WSGI)

WSGI is a simple calling convention for web servers to forward requests to web applications or frameworks written in the Python programming language.

* Python Egg

The `.egg` file is a distribution format for Python packages. It’s just an alternative to a source code distribution or Windows exe. But note that for pure Python, the .egg file is completely cross-platform.

* `setup.py`

`setup.py` is the python version of Java's `POM.xml` and c++'s `CMakeLists.txt`.

Some popular fields are as below, where `entry_points` is meta data that will be read at runtime (example use case is binding with `development.ini` (similar to Java's `config.properties`) that further describes the project with info such as `<host>:<port>` for a web server).

```python
from setuptools import setup

requires = [
    'pyramid',
    'waitress',
]

project_descr = """
This project is a tutorial
"""

setup(
    name='tutorial_proj',
    version='0.0.1',
    install_requires=requires,
    license='MIT',
    long_description=project_descr,
    entry_points={
        'paste.app_factory': [
            'main = tutorial_proj:main'
        ],
    },
)
```

The python project setup is run by `python setup.py install`.

* `__init__.py`

Python defines two types of packages, regular packages (Python 3.2 and earlier) and namespace packages. 

A regular package is typically implemented as a directory containing an `__init__.py` that is implicitly executed used to distinctly identify objects in a package's namespace.

## Config via `.ini`

