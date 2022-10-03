# Oracle knowledge

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