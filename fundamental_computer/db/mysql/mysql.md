# MYSQL

## MYSQL Engines

InnoDB is the default transactional storage engine for MySQL.

MyISAM is the default storage engine for MySQL.

### InnoDB Overview

InnoDB stores its data in a series of one or more data files that are collectively known as a *tablespace*.

It defaults to the REPEATABLE READ isolation level, and it has a
next-key locking strategy that prevents phantom reads in this isolation level (InnoDB locks gaps in the index structure)

### MyISAM Overview

MyISAM typically stores each table in two files: a data file and an index file.

It provides data redundancy backup, repair, logging, etc.

## Profiling

`SHOW STATUS` and `SHOW PROFILEs` to see query execution results, alternatively, same results can be obtained from `INFORMATION_SCHEMA.PROFILING`

### SHOW PROFILES

It returns query process total and sub-task durations:

```
mysql> SHOW PROFILES;
+----------+------------+---------------+
| Query_ID | Duration | Query           |
+----------+------------+---------------+
| 1 | 0.167679 | SELECT * FROM db.table |
+----------+------------+---------------+
```

then
```
mysql> SHOW PROFILE FOR QUERY 1;
+----------------------+----------+
| Status               | Duration |
+----------------------+----------+
| starting             | 0.000082 |
| Opening tables       | 0.000459 |
| System lock          | 0.000010 |
| Table lock           | 0.000020 |
| checking permissions | 0.000005 |
...
```

### SHOW STATUS

It returns a number of process counters:

```
mysql> SHOW STATUS WHERE Variable_name LIKE 'Handler%'
OR Variable_name LIKE 'Created%';
+----------------------------+-------+
| Variable_name              | Value |
+----------------------------+-------+
| Created_tmp_disk_tables    | 2     |
| Created_tmp_files          | 0     |
| Created_tmp_tables         | 3     |
| Handler_commit             | 1     |
| Handler_delete             | 0     |
...
```

### SHOW PROCESSLIST

This shows the number of running threads with respects to various functionalities.

### EXPLAIN

`EXPLAIN` runs after a SQL query to find detail engine operations.

## SQL Insights

* `ALTER TABLE`: making an empty table with the desired new structure, inserting all the data from the old table into the new one, and deleting the old table. Locks are applied to a varied degree.

* `WHERE`

• Apply the conditions to the index lookup operation to eliminate non-matching
rows. This happens at the storage engine layer.

• Use a covering index (“Using index” in the Extra column) to avoid row accesses,
and filter out non-matching rows after retrieving each result from the index. This
happens at the server layer, but it doesn’t require reading rows from the table.

• Retrieve rows from the table, then filter non-matching rows (“Using where” in the Extra column). This happens at the server layer and requires the server to read rows from the table before it can filter them.

## Replication

For scaling/high availability purposes, MySQL supports two kinds of replication: *statement-based replication* and *row-based
replication*.

Process:
1. The master records changes to its data in its binary log. (These records are called
binary log events.)
2. The replica copies the master’s binary log events to its relay log.
3. The replica replays the events in the relay log, applying the changes to its own data.

### Statement-based replication

Statement-based replication works
by recording the query that changed the data on the master. 
When the replica reads the event from the relay log and executes it, it is reexecuting the actual SQL query that the master executed.

### Row-based replication

Row-based replication records the actual data
changes in the binary log.
