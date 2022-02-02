# 8-bit single-cycle CPU

Reference: https://stanford.edu/~sebell/oc_projects/ic_design_finalreport.pdf

Here designs an 8-bit microprocessor:
* 8-bit data bus
* 8-bit address bus
* 8-bit ALU
* 8-bit generic registers
* 4-bit memory address register
* 16 bytes of RAM
* 4-bit program counter
* 4-level adjustable clocks (1Hz, 5Hz, 10Hz, 25Hz)
* 4-bit opcode

$\space \space$ Notes: 

LSR/LSL: Logical Shift Right/Left

ASR/ASL: Arithmetic Shift Right/Left

LD: Load

There's a register to hold information about the last operation computed by the ALU. This register is called the Condition Codes, usually abbreviated CC. 

Most typical are 
1) always (default result, such as `void` from C-compatible function); 
2) zero
3) carry (the current register width cannot hold a large number of too many bits, need to indicate concatenations as one result)
4) negative
5) overflow

...

## CPU modules

### Generic 8-bit registers

They can hold tmp 8-bit data for various purposes. Such as Stack pointer, temp ALU computation results, etc.

### Program counter

It indicates the step/process of a running program. 

For example, JUMP register may be triggered with some flag and directly sets a program counter to a specific value to load another instruction rather than the one next scheduled. 

### ALU

Arithmetic Logic Unit (ALU) is implemented to perform ADD, SUB, MUL, etc.

An ALU latch can be included for complex calculation purposes (large numbers of many bits). It grabs the result of the ALU operation, holds it, and then puts it on the databus when the store signals are asserted.

### Control module

Read from instruction register to determine next action on next clock cycle. Some most typical action signals are
* LOAD (fetch data from memory)
* STORE (write data into memeory)
* FETCH INSTRUCTION
* JMP

Besides such action signals, it controls delays/waits and optimizations for parallelism.

For example, When CPU receives `MOV eax 100`, it first loads it into an instruction register to be read. The following action include: increase Program Counter (pc) by 1, prepare a generic purpose register and fill it with the value 100. It also checks flag register to see if any flag set (such as Carry/Negative/Zero/...).

### Memory

A memory module takes inputs such as `in_addr` and `in_data` from modules such as ALU. A memory chip connects to ROMs and RAMs to perform data read and write operation.

## CPU DataPath

A datapath module is the description of how bits of data flow 