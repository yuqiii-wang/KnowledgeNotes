# Executable *ELF* (Executable and Linkable Format)

## Linux's Process on ELF

On start, Linux registers its supported binary format into `struct linux_binfmt` below.
```cpp
struct linux_binfmt {
	struct list_head lh;
	struct module *module;
	int (*load_binary)(struct linux_binprm *);
	int (*load_shlib)(struct file *);
	int (*core_dump)(struct coredump_params *cprm);
	unsigned long min_coredump;	/* minimal dump size */
} __randomize_layout;
```

ELF is a binary format registered to this `linux_binfmt` as below
```cpp
static struct linux_binfmt elf_format = {
	.module		= THIS_MODULE,
	.load_binary	= load_elf_binary,
	.load_shlib	= load_elf_library,
	.core_dump	= elf_core_dump,
	.min_coredump	= ELF_EXEC_PAGESIZE,
};
```
where `load_elf_binary` Linux kernel function runs checking the ELF binary content.

### Before `load_elf_binary`

From a system perspective of how to start a process:
* set up a virtual memory space to hold ELF content and for later instruction execution
* read ELF headers, and build a mapping relationship between the virtual memory and ELF
* set CPU instruction register to the entry of the ELF

Virtual memory setup is done with `execv()`/`execve()`. Depending on situations, it may fork a new virtual memory space or use existing parent virtual memory.

Eventually, `load_elf_binary()` gets invoked.

### Inside `load_elf_binary`

1. read elf headers
2. segment checking
3. find an interpreter
4. virtual mem mapping
5. get an elf entry
6. execute the elf instructions with a new thread 

```cpp
static int load_elf_binary (struct linux_binprm *bprm)
{
     // -------------------------------/
     /* Get and check the exec-header */
	loc->elf_ex = *((struct elfhdr *)bprm->buf);
	retval = -ENOEXEC;
	if (memcmp(loc->elf_ex.e_ident, ELFMAG, SELFMAG) != 0)
		goto out;
	if (loc->elf_ex.e_type != ET_EXEC && loc->elf_ex.e_type != ET_DYN)
          goto out;

     // -------------------------------/
     // a program must have at least one segment; one segment has size upto 65536U 
     /* Sanity check the number of program headers... */
	if (elf_ex->e_phnum < 1 ||
		elf_ex->e_phnum > 65536U / sizeof(struct elf_phdr))
		goto out;
        
	/* ...and their total size. */
	size = sizeof(struct elf_phdr) * elf_ex->e_phnum;
	if (size > ELF_MIN_ALIGN)
	goto out;
	
     // kernel mem allocation
	elf_phdata = kmalloc(size, GFP_KERNEL);
	if (!elf_phdata)
	goto out;
	/* Read in the program headers */
	retval = kernel_read(elf_file, elf_phdata, size, &pos);
	if (retval != size) {
		err = (retval < 0) ? retval : -EIO;
		goto out;
     }

     // -------------------------------/
     // find an interpreter (labeled as `.interp`), that is ued to load
     // dynamic/shared libraries, for example,
     // `/lib64/ld-linux-x86-64.so.2` is the dynamic linker for linux 64-bit ELF binary
     if (elf_ppnt->p_type == PT_INTERP) {
          elf_interpreter = kmalloc(elf_ppnt->p_filesz,    GFP_KERNEL);
	if (!elf_interpreter)
		goto out_free_ph;

	pos = elf_ppnt->p_offset;
	retval = kernel_read(bprm->file, elf_interpreter,  elf_ppnt->p_filesz, &pos);
	
	interpreter = open_exec(elf_interpreter);
     }

     // -------------------------------/
     // create a mapping via `elf_map()` to map virtual mem and object file
     for(i = 0, elf_ppnt = elf_phdata;  i < loc->elf_ex.e_phnum; i++, elf_ppnt++) {

          /* check virtual mem and page info  */ 
          if (elf_ppnt->p_flags & PF_R)
               elf_prot |= PROT_READ;
          if (elf_ppnt->p_flags & PF_W)
               elf_prot |= PROT_WRITE;
          if (elf_ppnt->p_flags & PF_X)
               elf_prot |= PROT_EXEC;
               
          vaddr = elf_ppnt->p_vaddr;

          error = elf_map(bprm->file, load_bias + vaddr, elf_ppnt,
                    elf_prot, elf_flags, total_size);

          // random bias is added to `load_bias` to prevent
          // hacking, that in i386, `.text` has static entry at `0x08048000`,
          // this is easy to be exploted by hackers
          if (elf_interpreter) {
               load_bias = ELF_ET_DYN_BASE;
               if (current->flags & PF_RANDOMIZE)
                    load_bias += arch_mmap_rnd();
               elf_flags |= elf_fixed;
          }

          // load mem by brk
          if ((current->flags & PF_RANDOMIZE) && (randomize_va_space > 1)) {
               current->mm->brk = 
               current->mm->start_brk =
               arch_randomize_brk(current->mm);
          }
     }

     // -------------------------------/
     // load interpreter and get elf_entry
     if (elf_interpreter) {
          unsigned long interp_map_addr = 0;

          elf_entry = load_elf_interp(&loc->interp_elf_ex,
		    interpreter,
		    &interp_map_addr,
		    load_bias, interp_elf_phdata);
	} else {
		elf_entry = loc->elf_ex.e_entry;
     }

     // -------------------------------/
     // register virtual mem
     current->mm->end_code = end_code;
     current->mm->start_code = start_code;
     current->mm->start_data = start_data;
     current->mm->end_data = end_data;
     current->mm->start_stack = bprm->p;

     if ((current->flags & PF_RANDOMIZE) && (randomize_va_space > 1)) {
          current->mm->brk = current->mm->start_brk =
     arch_randomize_brk(current->mm);
     }

     // -------------------------------/
     // start the process by a thread
     void start_thread(struct pt_regs *regs, unsigned long new_ip, unsigned long new_sp)
     {
          start_thread_common(regs, new_ip, new_sp,
               __USER_CS, __USER_DS, 0);
     }
}
```


## Sections and Segments

Reference:
https://docs.oracle.com/cd/E23824_01/html/819-0690/glcdi.html#scrolltoc

Executable and Linkable Format (*ELF*, formerly named Extensible Linking Format), is a common standard file format for executable files, object code, shared libraries, and core dumps, widely used in Unix distributions.

![Elf-layout--en.svg](imgs/Elf-layout--en.svg.png "Elf-layout--en.svg")

* Common applied extensions:

none, .axf, .bin, .elf, .o, .out, .prx, .puff, .ko, .mod, and .so

The segments contain information that is necessary for runtime execution of the file, while sections contain important data for linking and relocation. 
One segment can contain many sections.

### Section: 

tell the linker if a section is either:
* raw data to be loaded into memory, e.g. .data, .text, etc.
* formatted metadata about other sections, that will be used by the linker, but disappear at runtime e.g. .symtab, .srttab, .rela.text

### Segment: 

tells the operating system:
* where should a segment be loaded into virtual memory
* what permissions the segments have (read, write, execute).

`readelf -l <your-elf-binary>` show the mapping between sections and segments:

### Typical Sections

* .bss

Uninitialized data that contribute to the program's memory image. By definition, the system initializes the data with zeros when the program begins to run. The section occupies no file space, as indicated by the section type `SHT_NOBITS`.

* .data, .data1

Initialized data that contribute to the program's memory image.

* .dynsym

Dynamic linking symbol table. 

* .init

Executable instructions that contribute to a single initialization function for the executable or shared object containing the section. 

* .init_array

An array of function pointers that contributes to a single initialization array for the executable or shared object containing the section.

* .interp

The path name of a program interpreter. Use `readelf -l <your-elf-binary>` to find the interpreter.

* .rodata, .rodata1

Read-only data that typically contribute to a non-writable segment in the process image.

* .symtab

Symbol table, as Symbol Table Section describes. If the file has a loadable segment that includes the symbol table, the section's attributes include the `SHF_ALLOC` bit. Otherwise, that bit is turned off.

* .note

Information in the format described in Note Section.

* .fini

Executable instructions that contribute to a single termination function for the executable or shared object containing the section. 

* .text

The text or executable instructions of a program. `objdump -d <your-object-file>` can show assembly code from the binary.

## A `touch` binary example

Below is an example of ELF for the binary `touch`:

run `readelf` to display an executable information:

### Displaying the header of an ELF file

The ELF header defines the format of a binary file, instructing CPU how to read the file. Typically, there are endian, 32/64 bit, OS, etc.

```
readelf -h /usr/bin/touch

ELF Header:
  Magic:   7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00 
  Class:                             ELF64
  Data:                              2's complement, little endian
  Version:                           1 (current)
  OS/ABI:                            UNIX - System V
  ABI Version:                       0
  Type:                              DYN (Shared object file)
  Machine:                           Advanced Micro Devices X86-64
  Version:                           0x1
  Entry point address:               0x4270
  Start of program headers:          64 (bytes into file)
  Start of section headers:          98808 (bytes into file)
  Flags:                             0x0
  Size of this header:               64 (bytes)
  Size of program headers:           56 (bytes)
  Number of program headers:         13
  Size of section headers:           64 (bytes)
  Number of section headers:         30
  Section header string table index: 29
```

### Display information about the program headers

The program header table tells the system how to create a process image. 

```
readelf -l /usr/bin/touch

Elf file type is DYN (Shared object file)
Entry point 0x4270
There are 13 program headers, starting at offset 64

Program Headers:
  Type           Offset             VirtAddr           PhysAddr
                 FileSiz            MemSiz              Flags  Align
  PHDR           0x0000000000000040 0x0000000000000040 0x0000000000000040
                 0x00000000000002d8 0x00000000000002d8  R      0x8
  INTERP         0x0000000000000318 0x0000000000000318 0x0000000000000318
                 0x000000000000001c 0x000000000000001c  R      0x1
      [Requesting program interpreter: /lib64/ld-linux-x86-64.so.2]
  LOAD           0x0000000000000000 0x0000000000000000 0x0000000000000000
                 0x00000000000029c0 0x00000000000029c0  R      0x1000
  LOAD           0x0000000000003000 0x0000000000003000 0x0000000000003000
                 0x000000000000e061 0x000000000000e061  R E    0x1000
  LOAD           0x0000000000012000 0x0000000000012000 0x0000000000012000
                 0x0000000000004850 0x0000000000004850  R      0x1000
  LOAD           0x00000000000170d0 0x00000000000180d0 0x00000000000180d0
                 0x0000000000000fd0 0x00000000000011c8  RW     0x1000
  DYNAMIC        0x0000000000017b78 0x0000000000018b78 0x0000000000018b78
                 0x00000000000001f0 0x00000000000001f0  RW     0x8
  NOTE           0x0000000000000338 0x0000000000000338 0x0000000000000338
                 0x0000000000000020 0x0000000000000020  R      0x8
  NOTE           0x0000000000000358 0x0000000000000358 0x0000000000000358
                 0x0000000000000044 0x0000000000000044  R      0x4
  GNU_PROPERTY   0x0000000000000338 0x0000000000000338 0x0000000000000338
                 0x0000000000000020 0x0000000000000020  R      0x8
  GNU_EH_FRAME   0x0000000000014dcc 0x0000000000014dcc 0x0000000000014dcc
                 0x00000000000003f4 0x00000000000003f4  R      0x4
  GNU_STACK      0x0000000000000000 0x0000000000000000 0x0000000000000000
                 0x0000000000000000 0x0000000000000000  RW     0x10
  GNU_RELRO      0x00000000000170d0 0x00000000000180d0 0x00000000000180d0
                 0x0000000000000f30 0x0000000000000f30  R      0x1

 Section to Segment mapping:
  Segment Sections...
   00     
   01     .interp 
   02     .interp .note.gnu.property .note.gnu.build-id .note.ABI-tag .gnu.hash .dynsym .dynstr .gnu.version .gnu.version_r .rela.dyn .rela.plt 
   03     .init .plt .plt.got .plt.sec .text .fini 
   04     .rodata .eh_frame_hdr .eh_frame 
   05     .init_array .fini_array .data.rel.ro .dynamic .got .data .bss 
   06     .dynamic 
   07     .note.gnu.property 
   08     .note.gnu.build-id .note.ABI-tag 
   09     .note.gnu.property 
   10     .eh_frame_hdr 
   11     
   12     .init_array .fini_array .data.rel.ro .dynamic .got 
```

### Display Section details
```
readelf -S /usr/bin/touch

There are 30 section headers, starting at offset 0x181f8:
Section Headers:
  [Nr] Name              Type             Address           Offset
       Size              EntSize          Flags  Link  Info  Align
  [ 0]                   NULL             0000000000000000  00000000
       0000000000000000  0000000000000000           0     0     0
  [ 1] .interp           PROGBITS         0000000000000318  00000318
       000000000000001c  0000000000000000   A       0     0     1
  [ 2] .note.gnu.propert NOTE             0000000000000338  00000338
       0000000000000020  0000000000000000   A       0     0     8
  [ 3] .note.gnu.build-i NOTE             0000000000000358  00000358
       0000000000000024  0000000000000000   A       0     0     4
  [ 4] .note.ABI-tag     NOTE             000000000000037c  0000037c
       0000000000000020  0000000000000000   A       0     0     4
  [ 5] .gnu.hash         GNU_HASH         00000000000003a0  000003a0
       00000000000000ac  0000000000000000   A       6     0     8
  [ 6] .dynsym           DYNSYM           0000000000000450  00000450
       00000000000008e8  0000000000000018   A       7     1     8
  [ 7] .dynstr           STRTAB           0000000000000d38  00000d38
       000000000000044e  0000000000000000   A       0     0     1
  [ 8] .gnu.version      VERSYM           0000000000001186  00001186
       00000000000000be  0000000000000002   A       6     0     2
  [ 9] .gnu.version_r    VERNEED          0000000000001248  00001248
       0000000000000080  0000000000000000   A       7     1     8
  [10] .rela.dyn         RELA             00000000000012c8  000012c8
       0000000000001038  0000000000000018   A       6     0     8
  [11] .rela.plt         RELA             0000000000002300  00002300
       00000000000006c0  0000000000000018  AI       6    25     8
  [12] .init             PROGBITS         0000000000003000  00003000
       000000000000001b  0000000000000000  AX       0     0     4
  [13] .plt              PROGBITS         0000000000003020  00003020
       0000000000000490  0000000000000010  AX       0     0     16
  [14] .plt.got          PROGBITS         00000000000034b0  000034b0
       0000000000000010  0000000000000010  AX       0     0     16
  [15] .plt.sec          PROGBITS         00000000000034c0  000034c0
       0000000000000480  0000000000000010  AX       0     0     16
  [16] .text             PROGBITS         0000000000003940  00003940
       000000000000d712  0000000000000000  AX       0     0     16
  [17] .fini             PROGBITS         0000000000011054  00011054
       000000000000000d  0000000000000000  AX       0     0     4
  [18] .rodata           PROGBITS         0000000000012000  00012000
       0000000000002dcc  0000000000000000   A       0     0     32
  [19] .eh_frame_hdr     PROGBITS         0000000000014dcc  00014dcc
       00000000000003f4  0000000000000000   A       0     0     4
  [20] .eh_frame         PROGBITS         00000000000151c0  000151c0
       0000000000001690  0000000000000000   A       0     0     8
  [21] .init_array       INIT_ARRAY       00000000000180d0  000170d0
       0000000000000008  0000000000000008  WA       0     0     8
  [22] .fini_array       FINI_ARRAY       00000000000180d8  000170d8
       0000000000000008  0000000000000008  WA       0     0     8
  [23] .data.rel.ro      PROGBITS         00000000000180e0  000170e0
       0000000000000a98  0000000000000000  WA       0     0     32
  [24] .dynamic          DYNAMIC          0000000000018b78  00017b78
       00000000000001f0  0000000000000010  WA       7     0     8
  [25] .got              PROGBITS         0000000000018d68  00017d68
       0000000000000280  0000000000000008  WA       0     0     8
  [26] .data             PROGBITS         0000000000019000  00018000
       00000000000000a0  0000000000000000  WA       0     0     32
  [27] .bss              NOBITS           00000000000190a0  000180a0
       00000000000001f8  0000000000000000  WA       0     0     32
  [28] .gnu_debuglink    PROGBITS         0000000000000000  000180a0
       0000000000000034  0000000000000000           0     0     4
  [29] .shstrtab         STRTAB           0000000000000000  000180d4
       000000000000011d  0000000000000000           0     0     1
Key to Flags:
  W (write), A (alloc), X (execute), M (merge), S (strings), I (info),
  L (link order), O (extra OS processing required), G (group), T (TLS),
  C (compressed), x (unknown), o (OS specific), E (exclude),
  l (large), p (processor specific)
```