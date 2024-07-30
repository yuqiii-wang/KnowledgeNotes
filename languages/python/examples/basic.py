def fib(n):
    if n < 2:
        return n
    else:
        fib(n - 2) + fib(n - 1)

fib(10)
