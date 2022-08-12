# Preprocessing work

## Token-pasting operator `##` macro

It permits separate tokens to be joined into a single token. In the below example, `token##n` is combined as `#n` is parsed as `9`.

```cpp
// preprocessor_token_pasting.cpp
#include <stdio.h>
#define paster( n ) printf_s( "token" #n " = %d", token##n )
int token9 = 9;

int main()
{
   paster(9);
}
```