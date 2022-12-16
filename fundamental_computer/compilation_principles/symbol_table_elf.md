# ELF Symbol Table

Explained with this example:
```cpp
// symb_test.cpp
struct A {
    int val = 0xeeee;
} a;

struct B {
    int val = 0x1;
} b;

struct C {
    int val = 0x0;
} c;

int main() {
    c.val = a.val + b.val;
    return 0;
}
```


Compile by `g++ symb_test.cpp -o symb_test.o`,

Read header by `readelf -SW symb_test.o`
```

Section Headers:
  [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
  [ 0]                   NULL            0000000000000000 000000 000000 00      0   0  0
  [ 1] .interp           PROGBITS        0000000000000318 000318 00001c 00   A  0   0  1
  [ 2] .note.gnu.property NOTE            0000000000000338 000338 000020 00   A  0   0  8
  [ 3] .note.gnu.build-id NOTE            0000000000000358 000358 000024 00   A  0   0  4
  [ 4] .note.ABI-tag     NOTE            000000000000037c 00037c 000020 00   A  0   0  4
  [ 5] .gnu.hash         GNU_HASH        00000000000003a0 0003a0 000024 00   A  6   0  8
  [ 6] .dynsym           DYNSYM          00000000000003c8 0003c8 000090 18   A  7   1  8
  [ 7] .dynstr           STRTAB          0000000000000458 000458 00007d 00   A  0   0  1
  [ 8] .gnu.version      VERSYM          00000000000004d6 0004d6 00000c 02   A  6   0  2
  [ 9] .gnu.version_r    VERNEED         00000000000004e8 0004e8 000020 00   A  7   1  8
  [10] .rela.dyn         RELA            0000000000000508 000508 0000c0 18   A  6   0  8
  [11] .init             PROGBITS        0000000000001000 001000 00001b 00  AX  0   0  4
  [12] .plt              PROGBITS        0000000000001020 001020 000010 10  AX  0   0 16
  [13] .plt.got          PROGBITS        0000000000001030 001030 000010 10  AX  0   0 16
  [14] .text             PROGBITS        0000000000001040 001040 000185 00  AX  0   0 16
  [15] .fini             PROGBITS        00000000000011c8 0011c8 00000d 00  AX  0   0  4
  [16] .rodata           PROGBITS        0000000000002000 002000 000004 04  AM  0   0  4
  [17] .eh_frame_hdr     PROGBITS        0000000000002004 002004 00003c 00   A  0   0  4
  [18] .eh_frame         PROGBITS        0000000000002040 002040 0000f0 00   A  0   0  8
  [19] .init_array       INIT_ARRAY      0000000000003df0 002df0 000008 08  WA  0   0  8
  [20] .fini_array       FINI_ARRAY      0000000000003df8 002df8 000008 08  WA  0   0  8
  [21] .dynamic          DYNAMIC         0000000000003e00 002e00 0001c0 10  WA  7   0  8
  [22] .got              PROGBITS        0000000000003fc0 002fc0 000040 08  WA  0   0  8
  [23] .data             PROGBITS        0000000000004000 003000 000018 00  WA  0   0  8
  [24] .bss              NOBITS          0000000000004018 003018 000008 00  WA  0   0  4
  [25] .comment          PROGBITS        0000000000000000 003018 00002b 01  MS  0   0  1
  [26] .symtab           SYMTAB          0000000000000000 003048 000618 18     27  44  8
  [27] .strtab           STRTAB          0000000000000000 003660 0001f9 00      0   0  1
  [28] .shstrtab         STRTAB          0000000000000000 003859 00010c 00      0   0  1
Key to Flags:
  W (write), A (alloc), X (execute), M (merge), S (strings), I (info),
  L (link order), O (extra OS processing required), G (group), T (TLS),
  C (compressed), x (unknown), o (OS specific), E (exclude),
  l (large), p (processor specific)
```

Look at `[26] .symtab`:
* offset: $\mathtt{0x3048}=12360 \text{ bytes}$ 
* total size:  $\mathtt{0x618}=1560 \text{ bytes}$ 
* each symbol size: $\mathtt{0x18}=24 \text{ bytes}$ 
* number of symbols: $1560/24 = 65$

The 24-byte symbols have the below structure:
```cpp
typedef struct {
	Elf64_Word	st_name; // 4 B (B for bytes), unique reference
	unsigned char	st_info; // 1 B, higer four bits used for symbol binding,
                             // lower four bits used for symbol type
                             // Symbol Binding: LOCAL, GLOBAL, WEAK, etc
                             // Symbol type: NOTYPE, OBJECT, FUNC, SECTION, etc
	unsigned char	st_other; // 1 B, symbol visibility
	Elf64_Half	st_shndx; // 2 B, 
	Elf64_Addr	st_value; // 8 B
	Elf64_Xword	st_size; // 8 B
} Elf64_Sym; // total size = 24 B
```

Read `.symtab` symbols by `readelf -sW symb_test.o`
```bash
Symbol table '.symtab' contains 65 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND 
     1: 0000000000000318     0 SECTION LOCAL  DEFAULT    1 
     2: 0000000000000338     0 SECTION LOCAL  DEFAULT    2 
     3: 0000000000000358     0 SECTION LOCAL  DEFAULT    3 
     4: 000000000000037c     0 SECTION LOCAL  DEFAULT    4 
     5: 00000000000003a0     0 SECTION LOCAL  DEFAULT    5 
     6: 00000000000003c8     0 SECTION LOCAL  DEFAULT    6 
     7: 0000000000000458     0 SECTION LOCAL  DEFAULT    7 
     8: 00000000000004d6     0 SECTION LOCAL  DEFAULT    8 
     9: 00000000000004e8     0 SECTION LOCAL  DEFAULT    9 
    10: 0000000000000508     0 SECTION LOCAL  DEFAULT   10 
    11: 0000000000001000     0 SECTION LOCAL  DEFAULT   11 
    12: 0000000000001020     0 SECTION LOCAL  DEFAULT   12 
    13: 0000000000001030     0 SECTION LOCAL  DEFAULT   13 
    14: 0000000000001040     0 SECTION LOCAL  DEFAULT   14 
    15: 00000000000011c8     0 SECTION LOCAL  DEFAULT   15 
    16: 0000000000002000     0 SECTION LOCAL  DEFAULT   16 
    17: 0000000000002004     0 SECTION LOCAL  DEFAULT   17 
    18: 0000000000002040     0 SECTION LOCAL  DEFAULT   18 
    19: 0000000000003df0     0 SECTION LOCAL  DEFAULT   19 
    20: 0000000000003df8     0 SECTION LOCAL  DEFAULT   20 
    21: 0000000000003e00     0 SECTION LOCAL  DEFAULT   21 
    22: 0000000000003fc0     0 SECTION LOCAL  DEFAULT   22 
    23: 0000000000004000     0 SECTION LOCAL  DEFAULT   23 
    24: 0000000000004018     0 SECTION LOCAL  DEFAULT   24 
    25: 0000000000000000     0 SECTION LOCAL  DEFAULT   25 
    26: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS crtstuff.c
    27: 0000000000001070     0 FUNC    LOCAL  DEFAULT   14 deregister_tm_clones
    28: 00000000000010a0     0 FUNC    LOCAL  DEFAULT   14 register_tm_clones
    29: 00000000000010e0     0 FUNC    LOCAL  DEFAULT   14 __do_global_dtors_aux
    30: 0000000000004018     1 OBJECT  LOCAL  DEFAULT   24 completed.8061
    31: 0000000000003df8     0 OBJECT  LOCAL  DEFAULT   20 __do_global_dtors_aux_fini_array_entry
    32: 0000000000001120     0 FUNC    LOCAL  DEFAULT   14 frame_dummy
    33: 0000000000003df0     0 OBJECT  LOCAL  DEFAULT   19 __frame_dummy_init_array_entry
    34: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS symb_test.cpp
    35: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS crtstuff.c
    36: 000000000000212c     0 OBJECT  LOCAL  DEFAULT   18 __FRAME_END__
    37: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS 
    38: 0000000000003df8     0 NOTYPE  LOCAL  DEFAULT   19 __init_array_end
    39: 0000000000003e00     0 OBJECT  LOCAL  DEFAULT   21 _DYNAMIC
    40: 0000000000003df0     0 NOTYPE  LOCAL  DEFAULT   19 __init_array_start
    41: 0000000000002004     0 NOTYPE  LOCAL  DEFAULT   17 __GNU_EH_FRAME_HDR
    42: 0000000000003fc0     0 OBJECT  LOCAL  DEFAULT   22 _GLOBAL_OFFSET_TABLE_
    43: 0000000000001000     0 FUNC    LOCAL  DEFAULT   11 _init
    44: 00000000000011c0     5 FUNC    GLOBAL DEFAULT   14 __libc_csu_fini
    45: 0000000000000000     0 NOTYPE  WEAK   DEFAULT  UND _ITM_deregisterTMCloneTable
    46: 0000000000004000     0 NOTYPE  WEAK   DEFAULT   23 data_start
    47: 0000000000004014     4 OBJECT  GLOBAL DEFAULT   23 b
    48: 0000000000004018     0 NOTYPE  GLOBAL DEFAULT   23 _edata
    49: 00000000000011c8     0 FUNC    GLOBAL HIDDEN    15 _fini
    50: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND __libc_start_main@@GLIBC_2.2.5
    51: 0000000000004000     0 NOTYPE  GLOBAL DEFAULT   23 __data_start
    52: 0000000000000000     0 NOTYPE  WEAK   DEFAULT  UND __gmon_start__
    53: 0000000000004008     0 OBJECT  GLOBAL HIDDEN    23 __dso_handle
    54: 0000000000002000     4 OBJECT  GLOBAL DEFAULT   16 _IO_stdin_used
    55: 0000000000001150   101 FUNC    GLOBAL DEFAULT   14 __libc_csu_init
    56: 0000000000004020     0 NOTYPE  GLOBAL DEFAULT   24 _end
    57: 0000000000001040    47 FUNC    GLOBAL DEFAULT   14 _start
    58: 000000000000401c     4 OBJECT  GLOBAL DEFAULT   24 c
    59: 0000000000004010     4 OBJECT  GLOBAL DEFAULT   23 a
    60: 0000000000004018     0 NOTYPE  GLOBAL DEFAULT   24 __bss_start
    61: 0000000000001129    35 FUNC    GLOBAL DEFAULT   14 main
    62: 0000000000004018     0 OBJECT  GLOBAL HIDDEN    23 __TMC_END__
    63: 0000000000000000     0 NOTYPE  WEAK   DEFAULT  UND _ITM_registerTMCloneTable
    64: 0000000000000000     0 FUNC    WEAK   DEFAULT  UND __cxa_finalize@@GLIBC_2.2.5
```

Inside this `.symtab`, the object `a`'s (Symbol Num $59$), `b`'s (Symbol Num $47$) and `c`'s (Symbol Num $58$) binary can be found by
$$
\mathtt{0x3048} + 59 \times \mathtt{0x18} = \mathtt{0x35d0} = 13776 \text{ bytes}
\\
\mathtt{0x3048} + 47 \times \mathtt{0x18} = \mathtt{0x34b0} = 13488 \text{ bytes}
\\
\mathtt{0x3048} + 58 \times \mathtt{0x18} = \mathtt{0x35b8} = 13752 \text{ bytes}
$$

So that its binary texts given the compiled `symb_test.o` are 
* `hexdump -C -s13776 -n24 symb_test.o` for `a`:
```
000035d0  3d 01 00 00 11 00 17 00  10 40 00 00 00 00 00 00  |=........@......|
000035e0  04 00 00 00 00 00 00 00                           |........|
000035e8
```
where `10 40 00 00 00 00 00 00` is exactly the same as the `a`'s value `0000000000004010` in the `.symtab`

* `hexdump -C -s13488 -n24 symb_test.o` for `b`:
```
000034b0  36 01 00 00 11 00 17 00  14 40 00 00 00 00 00 00  |6........@......|
000034c0  04 00 00 00 00 00 00 00                           |........|
000034c8
```
where `14 40 00 00 00 00 00 00` is exactly the same as the `b`'s value `0000000000004014` in the `.symtab`

* `hexdump -C -s13752 -n24 symb_test.o` for `c`:
```
000035b8  0a 00 00 00 11 00 18 00  1c 40 00 00 00 00 00 00  |.........@......|
000035c8  04 00 00 00 00 00 00 00                           |........|
000035d0
```
where `1c 40 00 00 00 00 00 00` is exactly the same as the `c`'s value `000000000000401c` in the `.symtab`