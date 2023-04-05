# VHDL

VHSIC Hardware Description Language (VHDL) is a hardware description language (HDL).

## Simulation Tool

`logisim-evaluation` is a good tool employing VHDL as HDL for circuit simulation.

```bash
curl https://github.com/logisim-evolution/logisim-evolution/releases/download/v3.7.2/logisim-evolution_3.7.2-1_amd64.deb
```

install
```bash
sudo dpkg --install logisim-evolution_3.7.2-1_amd64.deb
```

run (you need java version 16+)
```bash
sudo -s
cd /opt/logisim-evolution/bin/
./logisim-evolution 
```

## Syntax

### Data types

`std_logic`: 0, 1, X, Z

`signal` vs `variable`: signal takes an assignment value after one clock cycle, while variable performs value assignment immediately

Process starting with `begin` blocks are the concurrent blocks. 

### Below defines an AND gate.

```vhdl
-- (this is a VHDL comment)
/*
    this is a block comment (VHDL-2008)
*/
-- import std_logic from the IEEE library
library IEEE;
use IEEE.std_logic_1164.all;

-- this is the entity
entity ANDGATE is
  port ( 
    I1 : in std_logic;
    I2 : in std_logic;
    O  : out std_logic);
end entity ANDGATE;

-- this is the architecture
architecture RTL of ANDGATE is
process (i_clock)
begin
  if rising_edge(i_clock) then
    O <= I1 and I2;
  end if;
end process;
end architecture RTL;
```
where `entity` is device declaration; `RTL` stands for `Register Transfer Level` design, `process` is equivalent to `always` in verilog.

`architecture` describes the behavior of an entity. 

`component` is used to reference `entity`, where `port` are arguments,
such as
```vhdl
component comp1 is
port (
  I1, I2: in std_logic;
  C: out std_logic
);
end component;
```

To use `component`, mapping is required.
```vhdl
comp: comp1 port map (
  ...
)
```

use `others` to init data
```vhdl
signal a : std_logic_vector (8 downto 0) := (others => '0');
```