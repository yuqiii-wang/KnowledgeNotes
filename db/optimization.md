# Optimizations

## Index design

Usually a max of 6 indices are used in a table

## `WHERE`, `GROUP BY` and `ORDER BY`

Every query's `WHERE` and `ORDER BY` should associate an index to avoid full table scan.

`WHERE` condition judgement should prevent using `!=` or `<>`, `NULL`, variables, function, or any sub query.

Should use `UNION` to include multiple `WHERE` conditions from different `SELECT` results rather than `OR` (`OR` might result in full table scan)

Use `EXIST` rather than `IN`

Filter out not used data rows before applying `GROUP BY`

## `DISTINCT`

Optimize `DISTINCT` with `GROUP BY`

## Use of procedure

Procedures are already optimized and stored in DB, hence it saved compilation/optimization time for a sql query.

## Data Type

Use small data type, such as `mediumint` is better than `int`

Use `enum` for options, such as `0, 1` to replace `female, male` of `VARCHAR`.

## Multiple tables and `JOIN`

SQL compiler takes long time to compile multiple `JOIN` statement.

There is a temp table in memory for each `JOIN` statement.

Building a temp table can help alleviate repeatedly scanning major tables. 

### Inner, left and right join

Use `INNER JOIN`, since it returns rows existed in both left and right tables

## Config

Enable cache, so that high frequency used queries are stored and returned fast.

Oracle compiler turns SQL lower case syntax words into capital before further action, so it is a good habit of writing SQL syntax words in capital 

Use `EXPLAIN` to find bottlenecks.