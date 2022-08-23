# Common DB knowledge

## DB Data Components

* Data: the contents of the database itself
* Metadata: the database schema
* Log Records: information about recent changes to the database
* Statistics: DB operation stats, such as data volume, number of data entries, etc.
* Indexes: data structure for fast query

## Query Evaluation Process

1) Parser: tokenization and build a parse tree, on which it validates if the SQL Query is good.

2) Preprocessor: check additional semantics for errors and access privileges

3) optimizer: based on existing statistics of such as number of affected rows, `WHERE` conditions (replaced with equivalent more efficient conditions), random access or sequential read, sequential `JOIN`s optimized into parallel `JOIN`s if results are the same.

* Manual optimization:

Manually set `HIGH_PRIORITY` and `LOW_PRIORITY` prior to executing transaction to dictate which transaction should run first.

`DELAYED` is used to hint delaying executing `INSERT` and `REPLACE`.

## Concepts

### Data model

data information about data structure (described by data schema), data operations (e.g., data query methods) and data constraints (e.g., if a field is nullable)

### Transaction

A transaction is a group of SQL queries that are treated
atomically, as a single unit of work. If the database engine can apply the entire group
of queries to a database, it does so, but if any of them can’t be done because of a crash
or other reason, none of them is applied. It’s all or nothing.

A transaction must satisfy a *ACID test*:

* Atomicity: function as a single indivisible unit of work so that the entire
transaction is either applied or rolled back.

* Consistency: before and after a transaction, the state of data should be consistent reflecting no changes when failed, and fully applied when succeeded.

* Isolation: transactions should have no interference/visibility on each other

* Durability: transaction change should be permanent 

### Isolation Level

* READ UNCOMMITTED: literally no locks applied, *dirty read* (read result undefined) happens when transaction change in progress

* READ COMMITTED: *unrepeatable read* happens when two transactions run in parallel, one of which read while another one `update` affecting at least one row of data.

* REPEATABLE READ: applied locks on affected existing rows (newly inserted rows are unaffected), however, *phantom read* happens when two transaction run in parallel, one of which read while another one `insert` affecting query result.

* SERIALIZABLE: applied locks on every row a transaction read; there is no transactions running in parallel.

### Event and routine

Code in periodic jobs is called events (events are store in `INFORMATION_SCHEMA.EVENTS` for MYSQL). Stored procedures and stored functions are collectively known as “stored
routines.”

*TRIGGER* lets you execute code when there’s an INSERT , UPDATE , or DELETE statement. You
can direct MySQL to activate triggers before and/or after the triggering statement executes.

### Cursor

*Cursors* are read-only because they iterate over temporary (in-memory) tables rather than the tables where the data originated. They let you iterate over query results row by row and fetch each row into variables for further processing.

Beware that a cursor only renders data recursively, a `SELECT` query still requires full computation time to get all result at once.

```sql
CREATE PROCEDURE bad_cursor()
BEGIN
    DECLARE film_id INT;
    DECLARE f CURSOR FOR SELECT film_id FROM filmdb.film;
    OPEN f;
    FETCH f INTO film_id;
    CLOSE f;
END
```

### Procedure

A SQL procedure is a  group of SQL statements and logic, compiled and stored together to perform a specific task.

```sql
Create  PROCEDURE GetStudentName 
(
    --Input parameter, employeID of the employee
    @employeeID INT,
    --Output parameter, employeeName of employee
    @employeName VARCHAR(50) OUT

    AS
    BEGIN
        SELECT @employeName = 
        Firstname + ' ' + Lastname 
        FROM Employee_Table 
            WHERE EmployeId=@employeID
    END
)
```

### View

A *view* is a virtual table that doesn’t store any data itself, acted as an "alias" to an ordinary query.

```sql
CREATE VIEW Oceania AS
    SELECT * FROM Country WHERE Continent = 'Oceania'
```

Benefits include
* Security - views are easy to manage access/execution permissions  
* Simplicity - views can be used to hide and reuse complex queries
* Maintenance - easy deployment of table structure changes
* Business-friendly - alias hide complex query logic and present to users as a simple view API

### Trigger

A SQL trigger is a database object which fires when an event occurs in a database. We can execute a SQL query that will "do something" in a database when a change occurs on a database table such as a record is inserted or updated or deleted. 

```sql
create trigger saftey
on database
for
create_table,alter_table,drop_table
as
print'you can not create, drop and alter table in this database'
rollback;
```

## Database storage structures

DB Storage with indexing:
ordered/unordered flat files, ISAM, heap files, hash buckets, or B+ trees

**Unordered** storage offers good insertion efficiency ( $O(1)$ ), but inefficient retrieval times ( $O(n)$ ). Most often, by using indexes on primary keys, retrieval times of $O(log n)$ or $O(1)$ for keys are the same as the database row offsets within the storage system.

**Ordered** storage typically stores the records in order. It has lower insertion efficiency, while providing more efficient retrieval of $O(log n)$

### Structured files

* Heap files

Heap files are lists of unordered records of variable size. New records are added at the end of the file, providing chronological order.

* Hash buckets

Hash functions calculate the address of the page in which the record is to be stored based on one or more fields in the record. So that given fields of a record (served as indexing), hash function can point to the memory address with $O(1)$.

* B+ trees

A B+ tree is an m-ary tree with a variable but often large number of children per node. B+ trees have very high fanout (number of pointers to child nodes in a node, typically on the order of 100 or more).

It searches with multiple branches, thus efficient
```py
def search(k):
    return tree_search(k, root)

def tree_search(k, node):
    if node is a_leaf:
        return node
    switch (k):
        case k ≤ k_0
            return tree_search(k, p_0)
        case k_i < k ≤ k_{i+1}
            return tree_search(k, p_{i+1})
        case k_d < k
            return tree_search(k, p_{d})
```

### Data Orientation

* "row-oriented" storage: 

each record/entry as a unit

* "column-oriented" storage: 

feature based storage

easy for data warehouse-style queries


## High Availability

Best way to optimize MySQL is to provide high performance hardware (still single instance, a.k.a. scaling up) and correct SQL query/schema for Mysql, after which, scaling out (provide more Mysql running machines) is required.

* Scaling Up: prefer multi-core CPU and multi-IO peripherals.

### Scaling Out:

* Replication: simply replicating instances from master db instance.

Might have limitations when high concurrency writes for data syncs.

* Functional partitioning, or division of duties: dedicate different nodes to different tasks.

This is about business level of providing db instances for targeted usage, such as one db for user login (db stores token, user profile, etc.) and another for user business services (db stores subscribed news, purchased products, etc.).

