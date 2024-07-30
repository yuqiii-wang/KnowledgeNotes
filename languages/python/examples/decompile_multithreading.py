import dis, marshal


pyc_path = "__pycache__/multi_threading.cpython-311.pyc"
with open(pyc_path, 'rb') as f:
    # First 16 bytes comprise the pyc header (python 3.6+), else 8 bytes.
    pyc_header = f.read(16)
    code_obj = marshal.load(f) # Suite to code object

dis.dis(code_obj)