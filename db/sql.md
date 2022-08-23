# Some SQL Knowledge

* SQL types

DDL (define, e.g., CREATE TABLE) 

DML (manipulation, e.g., CRUD)

DCL (Control, e.g., authority management)

* `SHOW`
```sql
SHOW DATABASES;
USE ${database_name};
SHOW TABLES;
SELECT * FROM ${table_name};
```

* `VARCHAR` vs `CHAR`

`CHAR` is a **fixed length** string data type, so any remaining space in the field is padded with blanks. CHAR takes up 1 byte per character. So, a CHAR(100) field (or variable) takes up 100 bytes on disk, regardless of the string it holds.

`VARCHAR` is a **variable length** string data type, so it holds only the characters you assign to it. `VARCHAR` takes up 1 byte per character, + 2 bytes to hold length information.  

* Prepared Statements

Database needs to parse sql scripts in advance to execution. By prepared statements, database caches parsed sql scripts to facilitate sql execution.

```sql
SET @sql := 'SELECT actor_id, first_name, last_name
-> FROM filmdb.actor WHERE first_name = ?';

PREPARE fetch_actor FROM @sql;

SET @actor_name := 'Penelope';

EXECUTE stmt_fetch_actor USING @actor_name;
```

* By default, sql lock is by row when `UPDATE`(write) and `SELECT`(read) conflict

* CHARSET

use `DEFAULT-CHARACTER-SET=UTF8MB4` (special emoji needs 4 bytes), both server and client need proper configuration.

In MySQL, edit `my.ini` to change database configuration.

In windows, find services.msc, find MySQL, find configuration files.

* `WITH`

`WITH` creates a context of using a temp table (this is a typical use case).

When a query with a `WITH` clause is executed, first the query mentioned within the clause is evaluated and the output of this evaluation is stored in a temporary relation. Following this, the main query associated with the WITH clause is finally executed that would use the temporary relation produced. 

The below code launches a tmp table `temporaryTable` referenced in the subsequent query.
```sql
WITH temporaryTable(averageValue) as
    (SELECT avg(Salary)
    from Employee)
        SELECT EmployeeID,Name, Salary 
        FROM Employee, temporaryTable 
        WHERE Employee.Salary > temporaryTable.averageValue;
```

* `CROSS APPLY` and `OUTER APPLY` (for SQL Server)

`CROSS APPLY` operator returns only those rows from the left table expression, similar to `INNER JOIN ` 

`OUTER APPLY` operator returns all the rows from the left table expression irrespective of its matches with the right table expression, similar to `LEFT OUTER JOIN`.