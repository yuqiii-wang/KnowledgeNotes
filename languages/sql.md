# Some SQL Knowledge

* VARCHAR vs CHAR

`CHAR` is a **fixed length** string data type, so any remaining space in the field is padded with blanks. CHAR takes up 1 byte per character. So, a CHAR(100) field (or variable) takes up 100 bytes on disk, regardless of the string it holds.

`VARCHAR` is a **variable length** string data type, so it holds only the characters you assign to it. VARCHAR takes up 1 byte per character, + 2 bytes to hold length information.  

