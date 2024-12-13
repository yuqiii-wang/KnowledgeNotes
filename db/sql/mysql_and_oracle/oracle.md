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

Oracle compiler turns SQL lower case syntax words into capital before further action, so it is a good habit of writing SQL syntax words in capital.

Use `EXPLAIN` to find bottlenecks.

## Sequence and Trigger

Index as `AUTO_INCREMENT` IN Oracle is done by setting up a `SEQUENCE` then trigger increment on update.

```sql
CREATE SEQUENCE product_id_seq
START WITH 1
INCREMENT BY 1
NOCACHE;
```

```sql
CREATE TABLE product (
    id NUMBER PRIMARY KEY,
    name VARCHAR2(255) NOT NULL
);
```

```sql
CREATE OR REPLACE TRIGGER product_trigger
BEFORE INSERT ON product
FOR EACH ROW
BEGIN
    IF :NEW.id IS NULL THEN
        :NEW.id := product_id_seq.NEXTVAL;
    END IF;
END;
```



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

## Others

### The `DUAL` table

`DUAL` in Oracle is a special single row table used for conduct non-query action for various purposes.

For example, to use query as connection/health check.

```sql
SELECT 1 FROM DUAL
```

To show now datetime.

```sql
SELECT SYSDATE AS current_datetime FROM DUAL;
```

When `MAX(PRODUCT.ID)` is not equal to `PRODUCT_ID_SEQ` (indicative of the auto increment by sequence is broken), use below to sync.
`DUAL` here is used as a virtual table for value assignment for `SELECT PRODUCT_ID_SEQ.NEXTVAL INTO curr_seq`.

```sql
DECLARE
  last_used NUMBER;
  curr_seq  NUMBER;
BEGIN
  SELECT MAX(PRODUCT.ID) INTO last_used FROM PRODUCT;
  LOOP
    SELECT PRODUCT_ID_SEQ.NEXTVAL INTO curr_seq FROM DUAL;
    IF curr_seq >= last_used THEN EXIT;
    END IF;
  END LOOP;
END
```
