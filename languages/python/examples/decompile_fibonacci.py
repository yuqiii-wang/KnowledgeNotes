import dis

fibonacci_code = '''
def fib(n): return n if n < 2 else fib(n - 2) + fib(n - 1)

fib(10)
'''

compiled_code = compile(fibonacci_code, '<string>', 'exec')
dis.dis(compiled_code)