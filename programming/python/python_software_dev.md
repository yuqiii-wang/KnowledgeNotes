# Python Software Development

## Python Software Setup and Build Basics

* Python Egg

The `.egg` file is a distribution format for Python packages same concept as a `.jar` to java or Windows `.exe`.
`.egg` file is completely cross-platform.

Essentially, it is a `.zip` with some metadata.

* `setup.py`

`setup.py` is the python version of Java's `POM.xml` and c++'s `CMakeLists.txt`.

The python project setup is run by `python setup.py install`.

* `__init__.py`

Python defines two types of packages, regular packages (Python 3.2 and earlier) and namespace packages.

A regular package is typically implemented as a directory containing an `__init__.py` that is implicitly executed used to distinctly identify objects in a package's namespace.

* Jinja2

*Jinja2* is a web template engine that combines a template with a certain data source to render dynamic web pages.

* The `python-config`

`python-config` is a command-line utility that provides information about the configuration of a Python installation.
The primary purpose of python-config is to make it easier for developers to compile and link programs or extensions that interact with Python.

Python uses it to locate compiled c++ include and libs folders.
Error might occur for incompatible libs vs python executable binary.

Check by

```sh
python-config --includes
python-config --libs

python --version
```

* Module not found

For example, `_ctypes` module is not found.

Check sys paths:

```sh
python -c "import sys; print(sys.path)"
```

Check where is `_ctypes.cpython-3xx-x86_64-linux-gnu.so` lib

```sh
python -c "import importlib.util; spec=importlib.util.find_spec('_ctypes'); print(spec)"
```

Likely it is in `path/to/python/lib/python3.xx/lib-dynload/`

## Poetry and `pyproject.toml`

## Type Safe by `pydantic`

`pydantic` is a Python library that provides runtime type checking and data validation using Python type annotations.

* `Field`

```py
from pydantic import BaseModel, validator, Field
from Typing import Optional

class Person(BaseModel):
    name: str = Field(default="Anonymous", description="The name of the person")
    age: int = Field(default=18, gt=0, description="The age of the person, be greater than 0")
    sex: Optional[str] = Field(default="Unknown", description="The sex of the person, either 'm', 'f', or 'Unknown'")

    @validator("sex")
    def validate_sex(cls, value):
        if value not in ["m", "f", "Unknown"]:
            raise ValueError("Sex must be either 'm', 'f', or 'Unknown'")
        return value

class Student(Person):
    student_id: str = Field(description="The unique ID of the student")
    courses: List[str] = Field(default_factory=list, description="List of courses the student is enrolled in")

#################
# Create a Student instance and test it
student_data = {
    "name": "Alice",
    "age": 20,
    "sex": "f",
    "student_id": "S12345",
    "courses": ["Math", "Physics"]
}

student = Student(**student_data)
print(student.model_dump())
```

## Unit Test by PyTest

Pytest is a popular, open-source testing framework for Python.

### PyTest Fixtures

PyTest Fixtures provides a consistent context where common use scenarios such as "run-before" and "run-after" are prepared.

In this example, the `sample_data` function is decorated with `@pytest.fixture` to mark it as a fixture.
Before `test_sum` and `test_prod` are executed, Pytest will call `sample_data` and pass its return value to the test function.
The `sample_data` is the "run-before" context for `test_sum` and `test_prod`.

```py
import pytest
import math

# A fixture to provide a sample list of numbers
@pytest.fixture
def sample_data():
    return [1, 2, 3, 4, 5]

# A test function that uses the sample_data fixture
def test_sum(sample_data):
    assert sum(sample_data) == 15

# A corrected test function that uses the sample_data fixture to test the product
def test_prod(sample_data):
    assert math.prod(sample_data) == 120
```

#### Scope in PyTest Fixtures

In Pytest, the scope parameter for a fixture determines its lifecycle—how often it is created and destroyed.

`scope="session"` and `scope="module"` are two common test lifecycle scopes:

* When a fixture is defined with `scope="module"`, it is set up once per test module (i.e., per `.py` file); typical use is loading read-only data.
* A fixture with `scope="session"` is the most long-lived. It is created only once for the entire test session, before any tests are run; typical use case is DB conn.

#### The `yield` in Fixtures

In `scope`, the fixture-annotated func is `yield`ed so that the context is retained.

For example, a db conn needs to be closed once done testing, the `yield conn` and temp store the context and resume once all tests are done.

```py
import pytest
import sqlite3

@pytest.fixture(scope="module")
def db_connection():
    # --- SETUP phase ---
    print("\n[Setting up DB connection...]")
    conn = sqlite3.connect(":memory:") # Create an in-memory database
    
    yield conn  # Provide the connection to the test and PAUSE

    # --- TEARDOWN phase (resumes here after test is done) ---
    print("\n[Tearing down DB connection...]")
    conn.close()

def test_database_write(db_connection):
    cursor = db_connection.cursor()
    cursor.execute("CREATE TABLE users (id INT, name TEXT)")
    cursor.execute("INSERT INTO users VALUES (1, 'Alice')")
    db_connection.commit()

    # Some assertion to make it a real test
    user = cursor.execute("SELECT name FROM users WHERE id=1").fetchone()
    assert user[0] == "Alice"
```

### The `pytest.ini`

The `pytest.ini` file serves as the primary configuration file for the `pytest` testing framework.

For example,

```ini
[pytest]

addopts = 
    -v
    --tb=short
    --color=yes
    -ra
    --no-cov

# Logging
log_cli = true
log_cli_level = INFO
log_cli_format = %(asctime)s [%(levelname)8s] %(message)s
log_cli_date_format = %Y-%m-%d %H:%M:%S
```

where

* `-v` (`--verbose`): Increases the verbosity of the output. 
* `--tb=short`: Controls the traceback style for failing tests. The short format is a cleaner and more concise representation of the error stack than the default
* `--color=yes`: This option forces the test output to be colorized
* `-ra`: The `-r` flag is for reporting, and the `a` character specifies which reports to show. `a` stands for "all" except for passes, delivered detailed non-passing tests.
* `--no-cov`: It disable code coverage test

The logging config prints log messages from application directly to the console during the test run.
