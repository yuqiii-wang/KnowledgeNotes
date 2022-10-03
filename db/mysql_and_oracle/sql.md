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

* `Null`

```sql
SELECT * FROM MyTable WHERE MyColumn != NULL (0 Results)
SELECT * FROM MyTable WHERE MyColumn <> NULL (0 Results)
SELECT * FROM MyTable WHERE MyColumn IS NOT NULL (568 Results)
```

`null` represents no value or an unknown value. 

`IS NULL` is specifically saying not comparing values, but rather it seeks out missing values.

`<>` and `!=` are often interchangeable.

* `COMMIT`

`COMMIT` is the SQL command that is used for storing changes performed by a transaction. 

When you perform a DML operation without a COMMIT statement, the changes are visible only to you.

Once applied `COMMIT`, there is no rollback. 

Oracle Database issues an implicit `COMMIT` before and after any data definition language (DDL) statement (such as `CREATE TABLE`).

By default, `COMMIT` has such arguments: `WORK WRITE IMMEDIATE WAIT`
1. `WORK` is only about SQL standard compliance, no additional impact on execution
2. `WRITE`
3. `IMMEDIATE` initiates I/O, causing the redo for the commit of the transaction to be written out immediately by sending a message to the LGWR process
4. `WAIT` ensures that the commit will not return until the corresponding redo is persistent in the online redo log.


* `ON DUPLICATE KEY UPDATE`

If you specify an `ON DUPLICATE KEY UPDATE` clause and a row to be inserted would cause a duplicate value in a `UNIQUE` index or `PRIMARY KEY`, an `UPDATE` of the old row occurs. For example, if column a is declared as `UNIQUE` and contains the value 1, the following two statements have similar effect:

```sql
INSERT INTO t1 (a,b,c) VALUES (1,2,3)
  ON DUPLICATE KEY UPDATE c=c+1;

UPDATE t1 SET c=c+1 WHERE a=1;
```

* *Window function* vs `GROUP BY`

```sql
SELECT country, SUM(profit) AS country_profit
       FROM sales
       GROUP BY country
       ORDER BY country;
```
that outputs
```bash
+---------+----------------+
| country | country_profit |
+---------+----------------+
| Finland |           1610 |
| India   |           1350 |
| USA     |           4575 |
+---------+----------------+
```

In contrast, window operations do not collapse groups of query rows to a single output row. Instead, they produce a result for each row. 

`SUM(profit) OVER(PARTITION BY country)` sums up profit per country. However, different from `GROUP BY`, window function does not collapses rows.

`OVER()` without argument, covers full table rows.

```sql
SELECT
         year, country, product, profit,
         SUM(profit) OVER() AS total_profit,
         SUM(profit) OVER(PARTITION BY country) AS country_profit
       FROM sales
       ORDER BY country, year, product, profit;
```
that outputs
```bash
+------+---------+------------+--------+--------------+----------------+
| year | country | product    | profit | total_profit | country_profit |
+------+---------+------------+--------+--------------+----------------+
| 2000 | Finland | Computer   |   1500 |         7535 |           1610 |
| 2000 | Finland | Phone      |    100 |         7535 |           1610 |
| 2001 | Finland | Phone      |     10 |         7535 |           1610 |
| 2000 | India   | Calculator |     75 |         7535 |           1350 |
| 2000 | India   | Calculator |     75 |         7535 |           1350 |
| 2000 | India   | Computer   |   1200 |         7535 |           1350 |
| 2000 | USA     | Calculator |     75 |         7535 |           4575 |
| 2000 | USA     | Computer   |   1500 |         7535 |           4575 |
| 2001 | USA     | Calculator |     50 |         7535 |           4575 |
| 2001 | USA     | Computer   |   1200 |         7535 |           4575 |
| 2001 | USA     | Computer   |   1500 |         7535 |           4575 |
| 2001 | USA     | TV         |    100 |         7535 |           4575 |
| 2001 | USA     | TV         |    150 |         7535 |           4575 |
+------+---------+------------+--------+--------------+----------------+
```