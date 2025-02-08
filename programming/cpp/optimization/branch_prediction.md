# Branch Prediction

## `likely` and `unlikely`

`__builtin_expect` provides the compiler with branch prediction information. 
User can add `likely` or `unlikely` to his/her condition testing if he/she has
already known the likely result of the condition. 
Compiler uses such information to optimize branching in assembly code generation.

```cpp
# define likely(x)	__builtin_expect(!!(x), 1)
# define unlikely(x)	__builtin_expect(!!(x), 0)
```

In the assembly code, `likely` and `unlikely` are translated into `jne` (jump not equal) and `je` (jump equal), respectively. 

Jump instruction ask CPU to fetch instructions from some other addresses, while without it, linear execution is fast (CPU can prefetch instructions).

### Example

Compile the code with `-O2` optimization. 
```cpp
#include <stdio.h>
#include <stdlib.h>
 
#define likely(x)    __builtin_expect(!!(x), 1)
#define unlikely(x)  __builtin_expect(!!(x), 0)
 
int main(int argc, char *argv[])
{
    int a  = atoi(argv[1]);
 
    if (likely(a==2)) {
        a++;
    } else {
        a--;
    }
 
    printf("%d\n", a);
 
    return 0;
}
```

Use `likely`
```x86asm
0000000000400510 <main>:
  ; atoi
  400510:	48 83 ec 08          	sub    $0x8,%rsp
  400514:	48 8b 7e 08          	mov    0x8(%rsi),%rdi
  400518:	ba 0a 00 00 00       	mov    $0xa,%edx
  40051d:	31 f6                	xor    %esi,%esi
  40051f:	e8 ec fe ff ff       	callq  400410 <strtol@plt>
 
  ; test a == 2 
  400524:	83 f8 02             	cmp    $0x2,%eax
 
  ; if a == 2 is true，run `a++` no jump
  ; if a != 2 is true, jump to 400541 to tun `a--`
  400527:	75 18                	jne    400541 <main+0x31>
  
  ; coz a == 2 is ture, gcc compiler optimizes `a++` to `0x3`
  400529:	be 03 00 00 00       	mov    $0x3,%esi
 
  ; run printf
  40052e:	bf 48 06 40 00       	mov    $0x400648,%edi
  400533:	31 c0                	xor    %eax,%eax
  400535:	e8 b6 fe ff ff       	callq  4003f0 <printf@plt>
  40053a:	31 c0                	xor    %eax,%eax
  40053c:	48 83 c4 08          	add    $0x8,%rsp
  400540:	c3                   	retq   
 
  // a--;
  400541:	8d 70 ff             	lea    -0x1(%rax),%esi
  400544:	eb e8                	jmp    40052e <main+0x1e>
 
  400546:	90                   	nop
```

If replaced with `unlikely`
```x86asm
0000000000400510 <main>:
  ; atoi
  400510:	48 83 ec 08          	sub    $0x8,%rsp
  400514:	48 8b 7e 08          	mov    0x8(%rsi),%rdi
  400518:	ba 0a 00 00 00       	mov    $0xa,%edx
  40051d:	31 f6                	xor    %esi,%esi
  40051f:	e8 ec fe ff ff       	callq  400410 <strtol@plt>
 
  ; test a == 2 
  400524:	83 f8 02             	cmp    $0x2,%eax
 
  ; when a == 2 jump to 40053f running `a++`
  ; when a!= 2 `a--`，no jump, no impact to CPU 
  400527:	74 16                	je     40053f <main+0x2f>
 
  ; a--
  400529:	8d 70 ff             	lea    -0x1(%rax),%esi
 
  ; run printf
  40052c:	bf 48 06 40 00       	mov    $0x400648,%edi
  400531:	31 c0                	xor    %eax,%eax
  400533:	e8 b8 fe ff ff       	callq  4003f0 <printf@plt>
  400538:	31 c0                	xor    %eax,%eax
  40053a:	48 83 c4 08          	add    $0x8,%rsp
  40053e:	c3                   	retq   
 
  ; a++;  coz `a == 2`, gcc compiler optimizes `a++` to `0x3`
  40053f:	be 03 00 00 00       	mov    $0x3,%esi
  400544:	eb e6                	jmp    40052c <main+0x1c>
  400546:	90                   	nop
```

## Prefetch

Prefetch is used to instruct CPU to prepare/early load data as addr informed in `prefetch_address`.

For GCC, it is declared as
```cpp
__builtin_prefetch((const void*)(prefetch_address),0,0);
```

`__builtin_prefetch` is clang/gcc specific. 
In x86 intrinsic `_mm_prefetch` is good with both clang and MSVC.

In modern CPU, prefetch is automatically performed such as in this scenario
```cpp
for(int i = 0; i < N; i++)
    for(int j = 0; j < M; j++)
        count += tab[i][j];
```

However, prefetch in work with cache would fail for this situation:
given a typical `CACHE_LINE_SIZE` of 64 bytes, there are 16 `CACHE_LINE_SIZE/sizeof(int)` units of integer per cache entry.

* Once `tab[i][0]` is read (after a cache miss, or a page fault), the data from `tab[i][0]` to `tab[i][15]` is copied to cache.
* When the code traverses in the row, i.e., `tab[i][M-1]` to `tab[i+1][0]`, it is highly likely to happen a cold cache miss, especially when `tab` is a dynamically-allocated array where each row could be allocated in a fragmented way.
Prefetch fails as a result of fragmented memory access.