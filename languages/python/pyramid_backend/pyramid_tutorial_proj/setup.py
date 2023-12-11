from setuptools import setup

# List of dependencies installed via `pip install -e .`
# by virtue of the Setuptools `install_requires` value below.
requires = [
    'pyramid',
    'waitress',
    'pyramid_chameleon',
    'pyramid_jinja2',
]

dev_requires = [
    'pyramid_debugtoolbar',
    'pytest',
    'webtest',
]

setup(
    name='pyramid_tutorial',
    version='0.0.1',
    install_requires=requires,
    license='MIT',
    extras_require={
        'dev': dev_requires,
    },
    py_modules = [
        'static',
        'views',
        'templates'
    ],
    entry_points={
        'paste.app_factory': [
            'main = pyramid_tutorial:main'
        ],
    },
)