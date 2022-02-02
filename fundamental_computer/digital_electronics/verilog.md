# Verilog

Verilog developed from the need to better simplify drawing of a complex circuit for fabrication on solicon. Basically, it describes how semiconductors are arranged and cooperated/wired to each other.

To install verilog
```bash
sudo apt update
sudo apt install verilog
```

Compile and run
```bash
# compile
iverilog -o <output_filename> <verilog_code_filename>

# simulation
vvp <output_filename>
```

### Verilog syntax and examples

* wire

`net_0`, `net_1`, `net_2`, ... represents a single wire denoted by indexing.

A group of wires form a vector as 
```v
module design;
    // 8-bit wire
    wire [7:0] addr;

    // assign bit value 1 to addr[0] single wire
    assign addr[0] = 1;
    // assign bit value 1 to addr[1] single wire
    assign addr[1] = 1;
    // assign bit value 0 to addr[2] single wire
    assign addr[2] = 0;

    // assign bit value 1111 to 4 wires addr[7:4]
    assign addr[7:4] = 4'hf;
endmodule
```

* datatype

`reg`: register result holding bits of data

`integer`: 32-bits var

`time`: unsigned 64-bits var usually for time tick

`real`: floating point value

```v
module testdata;
    integer int_a;
    real real_b;
    time time_c;

    initial begin
        int_a = 32'habcd_1234;
        real_b = 0.123456;
        time_c = $time;

        $display("a 0x%0h", int_a);
        $display("b %0f", real_b);
        $display("c %0t", time_c);
    end
endmodule
```

* string

Strings are truncated if too large to fit into a register. Each ASCII value needs one byte of space.

```v
// 11 bytes, no truncation
reg[8*11:1] str1 = "Hello World";

// 2 bytes, multiple truncations
reg[8*2:1] str2 = "Hello World";
```

* Memories

RAMs and ROMs are typical examples of memeories.
```v
// 16-bit vector 2D array with rowNum=4 and colNum=2
reg [15:0] mem [0:3][0:1];
```

* function

Function takes arguments as input/output ports.
```v
module xnor(input in_1, input in_2, output out);
    assign out = ~(in_1 ^ in_2); 
endmodule
```

* D flip-flops

A D flip-flop is a sequential element that follows the input pin d at the clock's given edge.

```v
// triggered when detecting positive edge of Clock or Reset signal
// `always` says sequential sync, without it, `posedge` is a one-time condition
always @(posedge Clock or posedge Reset)  
```