# Dynamic Programming

Wherever there is a recursive solution that has repeated calls for same inputs, it can be optimized using Dynamic Programming. 
The idea is to simply store the results of subproblems to avoid recomputation when needed later. 

For example,
Recursion solution for Fibonacci numbers:
```cpp
int fb(int n) {
    if (n <= 1) return n;
    else return fb(n-1) + fb(n-2);
}
```

It can be written as
```cpp
int fb(int n) {
    f[0] = 0;
    f[1] = 1;
    for (int i = 2; i <= n; i++) {
        f[i] = f[i-1] + f[i-2];
    }
    return f[n];
}
```