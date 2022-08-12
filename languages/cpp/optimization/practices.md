# Common optimization practices

## Memory access by block 

```cpp
// multiple threads running a lambda
int a[N];
std::thread* threadRunsPtrs[n];
for (int i = 0; i < n; i++)
{
    threadRunsPtrs[i] = new thread(lambdaPerThread, i, N/n);
}

// each thread accessing an individual block of mem,
// good for parallel computation
auto lambdaPerThread = [&](int threadId; int blockRange)
{
    for (int i = blockRange*threadId; i < blockRange*(threadId+1); i++)
    {
        a[i];
    }
}

// scattered mem access, bad for parallel computation
// actually only one thread is running at a time
auto lambdaPerThread = [&](int threadId; int threadTotalNum)
{
    for (int i = threadId; i < N; i += threadTotalNum)
    {
        a[i];
    }
}
```