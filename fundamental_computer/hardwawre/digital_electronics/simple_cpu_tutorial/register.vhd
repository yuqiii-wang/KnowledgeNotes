library ieee;

use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity register is
	port(
		i_clkin : in std_logic; -- clock in
		i_reset : in std_logic; -- reset in
		i_data_bus_in: in std_logic_vector (7 downto 0);  --For internal buses seperate buses are used for data in/out
		o_data_bus_out : out std_logic_vector (7 downto 0)
	);
end entity register;

architecture rtl of register is
	
	signal r_data_reg : std_logic_vector(7 downto 0) := (others => '0');  --Initialize register to zero
	
	begin
	
	p_load : process(i_clkin, i_reset) is
		begin
			if i_reset = '0' then --Asynchronus reset is used since clock can be halted which means synchronous reset won't work
				r_data_reg <= (others => '0');
			elsif rising_edge(i_clkin) then
                r_data_reg <= i_data_bus_in;  --Load the data from the bus in to the register
			end if;
	end process p_load;
		
	--Putting the data in the register on to the bus
	o_data_bus_out <= r_data_reg;
							
end rtl;
	
	
