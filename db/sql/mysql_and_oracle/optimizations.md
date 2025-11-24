# Optimizations

## Configs

To locate conf file:
```bash
/usr/sbin/mysqld --verbose --help | grep -A 1 'Default options'
```

Typically they are in `/etc/my.cnf` or `/etc/mysql/my.cnf`.

Run below to retrieve global config variables.
```sql
SHOW GLOBAL VARIABLES
```

Common configs are

* low_priority_updates: assign different level priority to `INSERT` , `REPLACE` , `DELETE` , and `UPDATE` queries against `SELECT` since read should have higher priority than write.

### Identify Bad Performance SQLs

Use `SHOW PROCESSLIST` to check the current ongoing SQLs as well as their elapsed time.
Locate the slow SQLs from the outputs.

Then use `SHOW PROFILE` OR `EXPLAIN` to further check in detail of the slow SQLs.

## `EXPLAIN`

|Columns|Descriptions|
|-|-|
|id|Represent a SELECT operation sequence|
|select_type|For example, SIMPLE means no JOIN or subquery|
|type|ALL: full table scan; INDEX: by index scan; range: index range scan|
|possible_keys|SQL optimizer considered index keys; not necessarily used though|
|key|SQL optimizer actually used keys; if NULL, no keys are used|

## Data Types

* Small: less space, better for cache, better for CPU concurrency

* Simple: primitive int, float, etc. rather than string; for example, `compare` takes one CPU cycle as against each char in a string

* no `null`: MYSQL engine interprets `null` with complex intermediate code

### Number

* `DECIMAL` supports exact math rather than `DOUBLE` for higher precision results.

* MYSQL supports int of different bit widths, such as `TINYINT` (2 bytes) and `BIGINT` (8 bytes)

* Use small data type, such as `mediumint` is better than `int`

* Use `enum` for options, such as `0, 1` to replace `female, male` of `VARCHAR`.


### String

* VARCHAR stores variable-length character strings, with 1 or 2 extra bytes to record the value’s length, e.g., `VARCHAR(1000)` can use up to 1002 bytes, actual used bytes are set at run time when actual data is placed.

* CHAR is fixed-length, padded with empty data for unused bytes.

* use `BLOB` and `TEXT` wisely 

### Datetime

* `DATETIME`: packed into an integer
in YYYYMMDDHHMMSS format, displayed such as 2008-01-16 22:37:08 by ANSI standard

* `TIMESTAMP` type stores the number of seconds elapsed since midnight, January 1, 1970, Greenwich Mean Time (GMT)

## Tricks

* UUID values, remove the dashes or, even better, convert the UUID values to 16-byte numbers with `UNHEX()` and store them in a `BINARY(16)` column

* Identifier (such as unique index), choose integers as identifier as they work with `AUTO_INCREMENT`, do not use string as it increases random access, creates memory splits

* Prefer Bulk insertion rather than one by one; because one row by one row insertion incurs too many connections and transactions

* MYSQL engine is row-major, so that if each row has many columns, it adds pressure on CPU for query/copy, etc.

* `LIMIT 1`: In business sometimes only need checking if a record exists by looking for the presence of at least one row of data. This scenario should use `LIMIT 1`, which prevents full table scan and returns immediately once a record is found.

* Use `UNION ALL`: by default, `UNION`'s results are applied `DISTINCT`. This results in additional duplicate row checking. If duplicate rows are allowed, use `UNION ALL` instead.

* Use `ORDER BY NULL`: by default, `GROUP BY`'s results are sorted; to prevent the default sorting, append `ORDER BY NULL` to the `GROUP BY` statement.

* reduce lock contention

* efficient caching

* take business into consideration (Schema Normalization, each fact is
represented once and only once, no duplicate data) for optimal read/write operation, for example,

|EMPLOYEE|DEPARTMENT|MANAGER|
|-|-|-|
|Jones|Accounting|Jones|
|Smith|Engineering|Smith|
|Brown|Engineering|Smith|
|Jack|Engineering|Smith|
|Zachary|Engineering|Smith|

if the manager of Engineering department (Smith) changes, there are four rows expecting changes and need complex SQL. If separating them as below, fewer rows are affected and need simple SQLs.

|EMPLOYEE|DEPARTMENT|
|-|-|
|Jones|Accounting|
|Smith|Engineering|
|Brown|Engineering|
|Jack|Engineering|
|Zachary|Engineering|

|DEPARTMENT|MANAGER|
|-|-|
|Accounting|Jones|
|Engineering|Smith|

* row order with indexing, good order makes engine run sequentially on one column scanning. `timestamp` is a typical good ordered index.

* minimal data access: only read data of interest

* Use many SQL queries (multiple transactions) than a simple heavy one. For example, required to delete 10,000 rows od data but not timely urgent, it can be chopped down to 100 rows per transaction.

* Aggregate functions such as `MAX`, `MIN` and `COUNT` are slow when `WHERE` is applied. One trick is to negate search conditions when the opposite data range is small. 

* Low-level hardware and OS:

Typically, they are 
RAID level, that higher levels require more disk; 
Use SSD hardware than mechanical; 
Keep frequent changing data entries in memory and write into disk later
