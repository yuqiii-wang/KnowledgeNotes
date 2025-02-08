#include </Users/yuqi/anaconda3/include/python3.12/Python.h>

// A simple C function to export
static PyObject* say_hello(PyObject* self, PyObject* args) {
    const char* name;
    if (!args || !PyArg_ParseTuple(args, "s", &name)) {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments passed to say_hello");
        return nullptr;
    }
    printf("Hello, %s!\n", name);
    Py_RETURN_NONE;
}

// Method definition
static PyMethodDef ModuleMethods[] = {
    {"say_hello", say_hello, METH_VARARGS, "Say hello to someone."},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

// Module definition
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "hello", // Module name
    nullptr,   // Module documentation
    -1,        // Module state size
    ModuleMethods
};

// Initialization function
PyMODINIT_FUNC PyInit_hello(void) {
    return PyModule_Create(&moduledef);
}
