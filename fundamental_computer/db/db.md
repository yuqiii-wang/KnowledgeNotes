# DB 

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

### Foreign Key vs Primary Key

*PRIMARY KEY* is used to identify a row entry, that must contain *UNIQUE* values, and cannot contain NULL values.

A *FOREIGN KEY* is a field (or collection of fields) in one table, that refers to the PRIMARY KEY in another table.

*FOREIGN KEY* requires server to do a lookup in another table when change happens such as `UPDATE`, to locate row entry addresses of the another table.

```sql
CREATE TABLE Persons (
    ID int NOT NULL,
    LastName varchar(255) NOT NULL,
    FirstName varchar(255),
    Age int,
    PRIMARY KEY (ID)
); 

CREATE TABLE Orders (
    OrderID int NOT NULL,
    OrderNumber int NOT NULL,
    PersonID int,
    PRIMARY KEY (OrderID),
    FOREIGN KEY (PersonID) REFERENCES Persons(PersonID)
); 
```

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

### Prepared SQL Statements

User can prepare SQL statements as below, that facilitates DB query, such as no need of repeatedly parse SQL.

```sql
SET @sql := 'SELECT actor_id, first_name, last_name
-> FROM filmdb.actor WHERE first_name = ?';

PREPARE fetch_actor FROM @sql;

SET @actor_name := 'Penelope';

EXECUTE stmt_fetch_actor USING @actor_name;
```