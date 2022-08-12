# SIMD

SIMD (Simple Instruction Multiple Data) is a specialized hardware component in CPU that performs parallel computation. It has dedicated SIMD registers to ALU.

An SSE register is 128 bit in size, and is named `__m128` if it is used to store four `float` (assumed each float has 4 bytes), or `__m128i` for `int` (assumed each int has 4 bytes). The AVX (Advanced Vector Extension) versions are `__m256` (‘octfloat’) and `__m256i` (‘octint’).  Similarly, `__m512` refers to data/array of 512 bits.

Depending on data type definition, for example for `__m512` working with `int`, `__m512` can either have 16 `int_32t`s or 8 `int_64t`s. `__m512` only represents SIMD register having 512 bits.

To check if your machine support SIMD:
```bash
grep -q sse2 /proc/cpuinfo && echo "SSE2 supported" || echo "SSE2 not supported"
grep -q avx2 /proc/cpuinfo && echo "AVX2 supported" || echo "AVX2 not supported"
```

### Common data type

|Data Type|Description|
|-|-|
|`__m128`   |128-bit vector containing 4 floats |
|`__m128d`	|128-bit vector containing 2 doubles|
|`__m128i`	|128-bit vector containing integers |
|`__m256`	|256-bit vector containing 8 floats |
|`__m256d`	|256-bit vector containing 4 doubles​|
|`__m256i`	|256-bit vector containing integers |

## Common SIMD APIs

* Load: copy data from mem to SIMD registers

|Intrinsic Name| Operation| Corresponding AVX Instructions|
|-|-|-|
|`_mm256_load_pd`| Load four double values, address aligned| VMOVAPD ymm, mem|
|`_mm256_loadu_pd`| Load four double values, address unaligned| VMOVUPD ymm, mem|

* Set: set register values with immediate numbers

|Intrinsic Name| Operation| Corresponding AVX Instructions|
|-|-|-|
|`_mm256_set_pd`| Set four values| Composite|
|`_mm256_setzero_pd`|Clear all four values to zeros|VXORPD|

* Store: 

|Intrinsic Name| Operation| Corresponding AVX Instructions|
|-|-|-|
|`_mm256_storeu_pd`|Store four values, addr unaligned|VMOVUPD|

* Arithmetics

|Intrinsic Name| Operation| Corresponding AVX Instructions|
|-|-|-|
|`_mm256_add_pd`| Addition |VADDPD|
|`_mm256_sub_pd`| Subtraction| VSUBPD|
|`_mm256_mul_pd`| Multiplication| VMULPD|
|`_mm256_div_pd`| Division| VDIVPD|
|`_mm256_sqrt_pd`| Squared Root |VSQRTPD|

* Comparisons

|Intrinsic Name| Operation| Corresponding AVX Instructions|
|-|-|-|
|`_mm256_cmp_pd`| Equal | VCMPPD|

* Conversion

|Intrinsic Name| Operation| Corresponding AVX Instructions|
|-|-|-|
|`_mm256_cvtepi32_pd`| Convert from 32-bit integer | VCVTDQ2PD |
|`_mm256_cvtpd_epi32`| Convert to 32-bit integer |VCVTPD2DQ|

* Shuffles