# Some Optimizations Over SQL

## Use of procedure

Procedures are already optimized and stored in DB, hence it saved compilation/optimization time for a sql query.

## Multiple tables and `JOIN`

Every `JOIN` statement needs one table scan of the to-be-joined tables.
Multiple `JOIN`s incur multiple scans of tables.

There is a temp table in memory for each `JOIN` statement.

Building a temp table can help alleviate repeatedly scanning major tables. 

## `LIMIT` Paging Performance Issues

The performance issue lies on row offset having **full row data return**.
For example, the below SQL finds rows between 300001 to 300005.

```sql
SELECT * FROM test WHERE val=4 LIMIT 300000,5; -- to search rows between 300001 and 300005.
```

SQL would return **all** rows from `SELECT * FROM test WHERE val=4` before reaching the 300005-th row met the condition `val=4`.
This puts burden on I/O.

Instead, include a subquery first selecting the index `id` then `JOIN` to the original table `test` aliased as `a`.
This only loads the `id` column from the `test` table for the whole 300005 rows of search.
```sql
SELECT * FROM test a INNER JOIN (SELECT id FROM test WHERE val=4 LIMIT 300000,5) b on a.id=b.id;
```

## Index design

Usually a max of 6 indices are used in a table

## `WHERE`, `GROUP BY` and `ORDER BY`

Every query's `WHERE` and `ORDER BY` should associate an index to avoid full table scan.

`WHERE` condition judgement should prevent using `!=` or `<>`, `NULL`, variables, function, or any sub query.

Should use `UNION` to include multiple `WHERE` conditions from different `SELECT` results rather than `OR` (`OR` might result in full table scan)

Use `EXIST` rather than `IN`

Filter out not used data rows before applying `GROUP BY`

