import dis

fibonacci_code = '''
def add(a,b): return a+b

add(1,2)
add(1.0, 2.0)
add("1", "2")
'''

compiled_code = compile(fibonacci_code, '<string>', 'exec')
dis.dis(compiled_code)

print("="*50)

fibonacci_code = '''
def add(a:int,b:int)->int: return a+b

add(1,2)
add(1.0, 2.0)
add("1", "2")
'''

compiled_code = compile(fibonacci_code, '<string>', 'exec')
dis.dis(compiled_code)

#### The typed `add(...)` func has additional checking.
#   2           2 LOAD_CONST               0 ('a')
#               4 LOAD_NAME                0 (int)
#               6 LOAD_CONST               1 ('b')
#               8 LOAD_NAME                0 (int)
#              10 LOAD_CONST               2 ('return')
#              12 LOAD_NAME                0 (int)
