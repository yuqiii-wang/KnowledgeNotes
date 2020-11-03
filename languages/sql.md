# Some SQL Knowledge

* Starter Scripts
```sql
SHOW DATABASES;
USE ${database_name};
SHOW TABLES;
SELECT * FROM ${table_name};
```

* VARCHAR vs CHAR

`CHAR` is a **fixed length** string data type, so any remaining space in the field is padded with blanks. CHAR takes up 1 byte per character. So, a CHAR(100) field (or variable) takes up 100 bytes on disk, regardless of the string it holds.

`VARCHAR` is a **variable length** string data type, so it holds only the characters you assign to it. VARCHAR takes up 1 byte per character, + 2 bytes to hold length information.  

* Prepared Statements

Database needs to parse sql scripts in advance to execution. By prepared statements, database caches parsed sql scripts to facilitate sql execution.

* By default, sql lock is by row when `UPDATE` and `SELECT`(read) conflict

* SQL types

DDL (define, e.g., CREATE TABLE) 

DML (manipulation, e.g., CRUD)

DCL (Control, e.g., authority management)

* STORE PROCEDURE

It is prefered when generating `JOIN`ed tables, rather than directly using sql scripts everytime sending signals from JDBC, called by `CALL PROCEDURE();`

* The Nature of `JOIN`

* CHARSET

use `DEFAULT-CHARACTER-SET=UTF8MB4` (special emoji needs 4 bytes), both server and client need proper configuration.

In MySQL, edit `my.ini` to change database configuration.

In windows, find services.msc, find MySQL, find configuration files.