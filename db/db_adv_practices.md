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

## Replication and Latency

MySQL supports two kinds of replication: *statement-based* replication and *row-based* replications.

MySQL slaves record changes in the master’s binary log 2 and replaying the log on the replica; the playback is an async operation.

There are I/O threads on loop detecting/fetching data from a master.

### Practices (for master to slave replication)

```sql
GRANT REPLICATION SLAVE, REPLICATION CLIENT ON *.*
TO repl@'192.168.0.%' IDENTIFIED BY 'p4ssword',;
```

On conf file, add (This makes MySQL synchronize the binary log’s contents to disk each time it commits a transaction, so you don’t lose log events if there’s a crash)
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

* `Slave_IO_State`, `SHOW ENGINE INNODB STATUS`, `Slave_SQL_Running_State` or `SHOW PROCESSLIST` – It tells you what the thread is doing. This field will provide you good insights if the replication health is running normally, facing network problems such as reconnecting to a master, or taking too much time to commit data which can indicate disk problems when syncing data to disk.

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

CTE is just a query block given defined table contexts. CTE can be fast since its operations are in RAM.  

Below code uses `temporaryTable` for all subsequent queries. 
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