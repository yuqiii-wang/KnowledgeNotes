# ASM (Assembly)

Read ASM from compiled c++ code by `g++ -S main.cpp -o main.asm`.

## asm declaration

asm-declaration gives the ability to embed assembly language source code within a C++ program.

## Basics

`mov dest src`

## Some c++ asm functions 

Due to portability issues, functions are compiler and OS specific.

### `spinlock`

Instead of context switches, a spinlock will "spin", and repeatedly check to see if the lock is unlocked. Spinning is very fast, so the latency between an unlock-lock pair is small. However, spinning doesn't accomplish any work, so may not be as efficient as a sleeping mutex if the time spent becomes significant.

```cpp
static inline unsigned xchg_32(void *ptr, unsigned x)
{
	__asm__ __volatile__("xchgl %0,%1"
				:"=r" ((unsigned) x)
				:"m" (*(volatile unsigned *)ptr), "0" (x)
				:"memory");

	return x;
}
```

```cpp
#define EBUSY 1
typedef unsigned spinlock;

static void spin_lock(spinlock *lock)
{
	while (1)
	{
		if (!xchg_32(lock, EBUSY)) return;
	
		while (*lock) cpu_relax();
	}
}

static void spin_unlock(spinlock *lock)
{
	barrier();
	*lock = 0;
}

static int spin_trylock(spinlock *lock)
{
	return xchg_32(lock, EBUSY);
}
```

### `cpu_relax()`

```asm
asm volatile("pause\n": : :"memory");
```
