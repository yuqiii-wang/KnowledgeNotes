# DB Advanced Practices

## Use of Lock

MySQL

* Table-level lock:

```sql
LOCK TABLE T READ;

LOCK TABLE T WRITE;
```

* Row-level lock

A shared (S) lock permits the transaction that holds the lock to read a row.

An exclusive (X) lock permits the transaction that holds the lock to update or delete a row.

MySQL (InnoDB) by default uses the REPEATABLE READ isolation level, and it has a
next-key locking strategy that prevents phantom reads in this isolation level (InnoDB locks gaps in the index structure)

* Page-level lock

A compromised page level is taken with overhead and conflict levels between row-level and table-level locking.

### Deadlock

When a deadlock is detected, InnoDB automatically rolls back a transaction.

### Optimistic vs. Pessimistic locking

* Optimistic locking: lock used only on row write/`UPDATE`, not used on read/`SELECT` (good performance, but inconsistency happens for simultaneous read and write)
* Pessimistic locking: lock used both on row write/`UPDATE`, and on read/`SELECT` (bad performance, but guaranteed consistency)

It is user-discretionary to say if read/write business is safe to apply optimistic or pessimistic lock.

For example, assume the below transactions are run in parallel multiple times.
It risks write skew.

```sql
START TRANSACTION;
SELECT * FROM test WHERE test.id > 10 AND test.id < 100;
UPDATE test SET val = 1 WHERE test.id > 10 AND test.id < 100;
```

The solution is by appending `FOR UPDATE` to `SELECT` (read is not available as the row might be undergoing `UPDATE`).
For index records the search encounters, `SELECT ... FOR UPDATE` locks the rows and any associated index entries.

```sql
START TRANSACTION;
SELECT * FROM test WHERE test.id > 10 AND test.id < 100 FOR UPDATE;
UPDATE test SET val = 1 WHERE test.id > 10 AND test.id < 100;
```

Similarly, there is `SELECT ... FOR SHARE`.
Other sessions can read in this mode, but to write have to wait until this session ends its transaction.

MySQL uses pessimistic locking by default.
Pessimistic locking uses `SELECT ... FOR UPDATE` that locks not only the rows, **but also the index** to produce REPEATABLE READ.

## Sub Query: Select-From-Where

Selected columns are by a WHERE condition which is derived from another select. For example:

```sql
select ename,deptno,sal
from emp
where deptno=(select deptno from dept where loc='NEW YORK')
```

## Transaction commit

### Explicit transactions

An explicit transaction is one in which you explicitly define both the start and end of the transaction through an API function or by issuing the Transact-SQL `BEGIN TRANSACTION`, `COMMIT TRANSACTION`, `COMMIT WORK`, `ROLLBACK TRANSACTION`, or `ROLLBACK WORK` Transact-SQL statements. When the transaction ends, the connection returns to the transaction mode it was in before the explicit transaction was started, either implicit or autocommit mode.

Some typical explicit transactions are 
* `CREATE DATABASE`
* `ALTER DATABASE`
* `DROP DATABASE`

### Implicit Transactions

When a connection is operating in implicit transaction mode, the instance of the SQL Server Database Engine automatically starts a new transaction after the current transaction is committed or rolled back. 

* `ALTER TABLE`
* `CREATE`
* `DELETE`
* `DROP`
* `FETCH`
* `GRANT`
* `INSERT`
* `OPEN`
* `REVOKE`
* `SELECT`
* `TRUNCATE TABLE`
* `UPDATE`

### Commit

Before `COMMIT`, DML changes are only visible to the user session. There is a dedicated storage area (called rollback area, prepared for rollback) associated with user session DML data changes.

Once applied `COMMIT`, the changes are permanent to DB, not in the user session lifecycle.

## Dirty Page and The `redo` Log



## Replication and Latency

MySQL supports two kinds of replication: *statement-based* replication and *row-based* replications.

MySQL slaves record changes in the master's binary log 2 and replaying the log on the replica; the playback is an async operation.

There are I/O threads on loop detecting/fetching data from a master.

### Practices (for master to slave replication)

```sql
GRANT REPLICATION SLAVE, REPLICATION CLIENT ON *.*
TO repl@'192.168.0.%' IDENTIFIED BY 'p4ssword',;
```

On conf file, add (This makes MySQL synchronize the binary log's contents to disk each time it commits a transaction, so you don't lose log events if there's a crash)

```cfg
sync_binlog=1
```

### Master to master replication

The default delay for master to master replication is $0$ seconds, and this can be config by the below statement.

```sql
CHANGE MASTER TO MASTER_DELAY = N;
```

### Latency

About debugging latency in replication:

* `Slave_IO_State`, `SHOW ENGINE INNODB STATUS`, `Slave_SQL_Running_State` or `SHOW PROCESSLIST` - It tells you what the thread is doing. This field will provide you good insights if the replication health is running normally, facing network problems such as reconnecting to a master, or taking too much time to commit data which can indicate disk problems when syncing data to disk.

* `Master_Log_File` - check bin log

* Use linux tools such as `ps`, `top`, `iostat`


## Compare *Temp Table*, *Table Variable* and *CTE* (Common Table Expression)

### Temp table

Temporary tables are similar to ordinary persistent tables, except that tmp tables are stored in `Tempdb`.

`Tempdb` typically resides in RAM, but it can be placed in disk if large. `Tempdb` performance tuning can improve performance.

Local `Tempdb`s (with a prefix `#`) are automatically dropped when the current user logout his/her session; global `Tempdb`s (with a prefix `##`) are automatically dropped when all users logout.

```sql
CREATE temporary TABLE tmp
(
id INT unsigned NOT NULL,
name VARCHAR(32) NOT NULL
)
engine=memory; -- change engine type if required e.g myisam/innodb
```

### CTE (Common Table Expression)

CTE is just a query block given defined table contexts. 
CTE can be fast since its operations are in RAM.

Below code uses `temporaryTable` for all subsequent queries. 
`temporaryTable` has the limited scope indicated by `WITH` defined as `(SELECT avg(Salary) from Employee)`, whose results are persistent throughout the `WITH` scope.

```sql
WITH temporaryTable(averageValue) as
    (SELECT avg(Salary)
    from Employee)
        SELECT EmployeeID,Name, Salary 
        FROM Employee, temporaryTable 
        WHERE Employee.Salary > temporaryTable.averageValue;
```

### Table Variable

Functionality similar to temp table, but it is a variable, which means no logs, no lock and no non-clustering indexing, as well as many other restrictions. 

It is used when hardware limitations are a bottleneck, since table variable uses less resources.

```sql
DECLARE @News Table 
　　( 
　　News_id int NOT NULL, 
　　NewsTitle varchar(100), 
　　NewsContent varchar(2000), 
　　NewsDateTime datetime 
　　)
```

### Use example

`T` is filled with many rows of `(A,B,C)`, where `A` is an incremental index; `B` is a random `int` col; `C` holds large texts.

```sql
CREATE TABLE T(
    A INT IDENTITY PRIMARY KEY, 
    B INT , 
    C CHAR(8000) NULL
    );
```

Here the sql query first finds rows where `B % 100000 = 0` stored as `T1` and `T2`, then find rows on `T2.A > T1.A`.

Below is a CTE example. This can be expensive since `T` might be scanned multiple times.

```sql
WITH CTETmp
     AS (SELECT *,
                ROW_NUMBER() OVER (ORDER BY A) AS RN
         FROM   T
         WHERE  B % 100000 = 0)
SELECT *
FROM   CTETmp T1
       CROSS APPLY (SELECT TOP (1) *
                    FROM   CTETmp T2
                    WHERE  T2.A > T1.A
                    ORDER  BY T2.A) CA 
```

Below is a temp table example, and this should be fast for `T` is only scanned once and the results are stored in a tempdb.

```sql
INSERT INTO #T
SELECT *,
       ROW_NUMBER() OVER (ORDER BY A) AS RN
FROM   T
WHERE  B % 100000 = 0

SELECT *
FROM   #T T1
       CROSS APPLY (SELECT TOP (1) *
                    FROM   #T T2
                    WHERE  T2.A > T1.A
                    ORDER  BY T2.A) CA 
```

## Bulk Insertion and Performance Optimization

