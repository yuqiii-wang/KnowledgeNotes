# 8-bit single-cycle CPU

Reference: https://stanford.edu/~sebell/oc_projects/ic_design_finalreport.pdf

Here designs an 8-bit microprocessor:
* 8-bit data bus
* 16-bit address bus
* Eight 8-bit registers
* A set of self-designed instructions and coresponding formats

Abbrvs: 
reg (register), 
cond (condition),
imm (immediate number)

Instruction Format:

| 15 - 11 | 10 - 9 | 8 | 7 - 5 | 4 - 2 | 1 - 0 |
| ------- | ------ | - | ----- | ----- | ----- |
| Opcode  | src type | dest type | src reg | dest reg | cond |

Instruction Set (Opcode)

| Instruction | Opcode | Operands |
| ----------- | ------ | -------- |
| NOP | 0x00 |  $\space$    |
| ADD | 0x01 | reg, reg/imm |
| SUB | 0x02 | reg, reg/imm |
| MUL | 0x03 | reg, reg/imm |
| AND | 0x04 | reg, reg/imm |
| OR  | 0x05 | reg, reg/imm |
| SHIFTRIGHT | 0x06 | reg   |
| SHIFTLEFT  | 0x07 | reg   |
| LD  | 0x08 | reg, imm/addr|
| MV  | 0x09 | addr, addr   |
| JMP  | 0x0A | addr, cond   |

