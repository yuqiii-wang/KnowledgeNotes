# Oracle knowledge

Oracle vs MySQL

|Oracle|MySQL|
|-|-|
|Proprietary, expensive|Free|
|Enterprise friendly (good admin management, cyber security, snapshot and backup, etc)|Community friendly (supported running in less popular OS such as Symbian)|
|Temp table: explicit deletion required|Temp table: deleted once a session ends|
|Variables: supported `CHAR`, `NCHAR`, `VARCHAR2` and `NVARCHAR2;`|Variables: supported `CHAR` and `VARCHAR`|
|Show some lines: `LIMIT`|Show some lines: `ROWNUM`|
|Other minor tech diffs...|Other minor tech diffs...|

## Redo log

In the Oracle RDBMS environment, *redo log*s comprise files in a proprietary format which log a history of all changes made to the database. 

A redo record, also called a redo entry, holds a group of change vectors, each of which describes or represents a change made to a single block in the database.

For example, if a user `UPDATE`s a salary-value in a table containing employee-related data, the DBMS generates a redo record containing change-vectors that describe changes to the data segment block for the table. And if the user then COMMITs the update, Oracle generates another redo record and assigns the change a "system change number" (SCN).

## Common oracle c++ interfaces

```cpp
typedef RWCollectable Object;  // Smalltalk typedef
#include <rw/collect.h>
```

## Config for Better Performance

Enable cache, so that high frequency used queries are stored and returned fast.

Oracle compiler turns SQL lower case syntax words into capital before further action, so it is a good habit of writing SQL syntax words in capital 

Use `EXPLAIN` to find bottlenecks.

## Function

Syntax:
```sql
FUNCTION function_name  
   [ (parameter [,parameter]) ]  
RETURN return_datatype  
IS | AS  
 [declaration_section]  
BEGIN  
   executable_section  
[EXCEPTION  
   exception_section]  
END [function_name];  
```

Example:
```sql
FUNCTION computeCashFlow (
    deposit NUMBER,
    interestRate NUMBER,
    totalPeriod NUMBER
) RETURN NUMBER
AS
    interest NUMBER;
    totalReturn NUMBER;
BEGIN
    interest := deposit * interestRate/100;
    totalReturn := deposit + interest;

    RETURN totalReturn;
END computeCashFlow
```

## Loop

```sql
DECLARE 
    i NUMBER := 0;
BEGIN
    LOOP
        i := i + 1;
        IF i > 5 THEN
            EXIT ;
        END IF;
    END LOOP;
END;
```

