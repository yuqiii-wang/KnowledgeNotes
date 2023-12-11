# install essentials for prod run
pip install -e .

# install dev tools for development and debug
pip install -e ".[dev]"

# start pyramid server
pserve ./development.ini --reload

# run unit tests
pytest unit_tests/tests.py -q