entity alu is
    port (
        i_eo : in std_logic;  --Enable output result from o_alu_bus_out (active when low)
		i_a : in std_logic_vector (7 downto 0); -- primary operand
		i_b : in std_logic_vector (7 downto 0); -- secondary operand
        i_opcode : in std_logic_vector (3 downto 0); 
		o_jmp_code : in std_logic_vector (1 downto 0); -- o_jmp_code[0]:carry, o_jmp_code[1]:zero
		o_alu_bus_out : out std_logic_vector (7 downto 0)
    );
end entity;

architecture rtl of alu is

    -- init all bits to zeros
	signal w_a : std_logic_vector (7 downto 0) := (others => '0');
	signal w_b : std_logic_vector (7 downto 0) := (others => '0');
	signal w_alu_out : std_logic_vector (8 downto 0) := (others => '0');

    --Instructions (OP codes)
	subtype t_instruction is std_logic_vector(3 downto 0);
	constant c_NOP  : t_instruction := "0000"; 	--No OPeration
	constant c_ADD  : t_instruction := "0010"; 	--ADDition
	constant c_SUB  : t_instruction := "0011";	--SUBtraction

	variable v_carry : std_logic;
	variable v_zero : std_logic;

	begin

    -- input
    w_a(7 downto 0) <= i_a;
    w_b(7 downto 0) <= i_b;
	v_carry = o_jmp_code(0);
	v_zero = o_jmp_code(1);

    case i_opcode is
		when c_NOP => 
			w_alu_out := w_b;
			v_carry := 0;
		when c_ADD => 
			w_alu_out := w_a + w_b;
			v_carry := w_alu_out(7);
		when c_SUB => 
			w_alu_out := w_a + w_b;
			v_carry := w_alu_out(7);

	--Putting the ALU result onto the bus
	o_alu_bus_out <= w_alu_out(7 downto 0) when(i_eo = '0') else
						  (others => '0');

end architecture rtl;