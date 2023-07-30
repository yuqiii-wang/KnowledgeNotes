# Sharding

## Partitioning vs Sharding

### Partitioning

By splitting a large table into smaller, individual tables, queries that access only a fraction of the data can run faster because there is less data to scan.

* Vertical partitioning: 
which splits a table's columns into two or more tables. 
It involves dividing the columns of a table and placing them in two or more tables linked by the original table's primary key.

* Horizontal partitioning (often called sharding): 
which splits a table's records into two or more tables. 
It involves splitting the rows of a table between two or more tables with identical structures.

### Sharding

Sharding is a special case of partitioning, a horizontal partition of data in a database.

Horizontal partitioning is a database design principle whereby rows of a database table are held separately (vertical partitioning is for column held data).

Each partition forms part of a shard, which may in turn be located on a separate database server or physical location. 

One typical partition key is date-based timestamp (Unix date count starting from 1970.01.01), 
that data entries generated from the same date are grouped into a shard (contiguous disk addresses).

<div style="display: flex; justify-content: center;">
      <img src="imgs/data_sharding_difficulty_level.png" width="60%" height="30%" alt="data_sharding_difficulty_level" />
</div>
</br>

## AWS Redshift and Sharding

Amazon Redshift is built on top of PostgreSQL 8.

Columnar storage for database tables drastically reduces the overall disk I/O requirements.

The smallest unit of IO the database can work with is a block (a block is a small size file on disk, 1 MB by default).
When `INSERT`/`UPDATE` happens, DB locks the affected block files then updates the block files, then releases the lock. 

Redshift assigns each file an internally assigned `filenode` number. 
User can query the catalog tables to find out the file name, but Redshift won’t allow you access to the underlying storage to verify if the file really exists on disk.

```sql
select relname, relfilenode
from pg_catalog.pg_class
where relname = '<table_name>';
```

For column-based DB, blocks contain column values instead of full rows.

### Block Meta Data

When Redshift writes blocks, it stores a bit of metadata for each block. 
A vital piece of information stored is the minimum and maximum value in each block, creating zone maps. 

Zone maps can speed up queries drastically by allowing Redshift to skip whole blocks while scanning for data.

For example, for a table `<table_name>` whose first col is of datetime.

```sql
select
  blocknum as block,
  minvalue as min,
  maxvalue as max
from stv_blocklist, stv_tbl_perm
where stv_blocklist.tbl = stv_tbl_perm.id
  and stv_blocklist.slice = stv_tbl_perm.slice
  and stv_tbl_perm.name = '<table_name>'
  and stv_blocklist.col = 1 /* the first col */
order by blocknum
```

By the above query, below results show how each block file stores ranges of datetime data.
This mapping is also called *zone map*.

|block | min | max |
|-|-|-|
|0 | 2014-1-1 | 2014-2-3 |
|1 | 2014-2-3 | 2014-3-8 |
|... | ... | ... | 
|1500 | 2018-1-1 | 2018-1-11 |
|1501 | 2018-1-11 | 2018-1-25|

### Distribution Key

Client sends a SQL query to DB's leader, that parses the SQL by distribution key to see which compute node should receive the query.
Inside each compute node, slice represents partially allocated computer resources: CPU cores and memory.

              Client
                |
                |
              Leader
      ---------------------
    /           |          \
 | Compute | Compute | Compute |
 | ------- | ------- | --------|
 | Slice 0 | Slice 2 | Slice 4 |
 | Slice 1 | Slice 3 | Slice 5 |

When a table is created in Redshift, the user can optionally choose a “distkey” for that table.
If no distkey is chosen, then the table is evenly distributed over all nodes.

Only one column in a table can be the distribution key.

When you load data into a table, Amazon Redshift distributes the rows to each of the compute nodes according to the table’s *DISTSTYLE*.
Within each compute node, the data are assigned to a *cluster slice* to process.

Key distribution styles:
* All: A copy of the entire table is distributed to every node. 
* Even: The leader node distributes the rows across the slices in a round-robin fashion, regardless of the values in any particular column. 
* Key: The rows are distributed according to the values in one column. The leader node places matching values on the same node slice.

### Sort Key

When you create a table, you can alternatively define one or more of its columns as sort keys. 
When data is initially loaded into the empty table, the rows are stored on disk in sorted order.

User can specify one or more columns as sort key columns for the table by using the SORTKEY (`column_name [, ...]`) syntax (since each block file stores only one col data, there can be many sort keys that only have impact on the sorted col, not others; in stark contrast to row-based DB where there is only one physical layout across all rows). 
Only compound sort keys are created with this syntax.

* Reduce disk I/O by improving zone map effectiveness.
* Reduce compute overhead and I/O by avoiding or reducing cost of sort steps.
* Improve join performance by enabling MERGE JOIN operation

### Sort Key: Compound vs Interleaved

* Compound:
looks just like the result of an `order by` on multiple columns.

Cols have priority, such as below where `customer_id` is prioritized. 

If by `SELECT * FROM <table_name> WHERE customer_id == 1`, DB only needs to look at 1 block;

If by `SELECT * FROM <table_name> WHERE product_id == 1`, DB needs to search in many blocks.

|block | customer_id | product_id | 
|-|-|-|
| 1 | 1 | 1 |
|1 | 1 | 2 |
|1 | 1 | 3 |
|1 | 1 | 4 |
|2 | 2 | 1 |
|2 | 2 | 2 |
|2 | 2 | 3 |
|2 | 2 | 4 |

* Interleaved:
every col has equal weight.

This is useful for `SELECT * FROM <table_name> WHERE customer_id == 1 AND product_id == 1`

|block | customer_id | product_id | 
|-|-|-|
| 1 | 1 | 1 |
|1 | 1 | 2 |
|1 | 2 | 1 |
|1 | 2 | 2 |
|2 | 1 | 3 |
|2 | 1 | 4 |
|2 | 2 | 3 |
|2 | 2 | 4 |

### Primary Key

A primary key implies that other tables can rely on this set of columns as a unique identifier for rows.

`PRIMARY KEY` columns are also defined as `NOT NULL`.

### Durability and Fault Tolerance

Build an additional pipeline to S3.

Build copy replica of compute nodes.

### Concurrency Inserts

In Redshift, the `INSERT` statement inserts a single row, while the `COPY` command loads data in bulk. 
Both can participate in a transaction.

Blocks are immutable by design, this means that for each statement executed on a single row, the database will need to clone a whole 1MB block.
During the clone then update then delete old block action, the affected range of data is on lock.

Immutable blocks allow Redshift to keep serving consistent reads while writes are happening. 
This is a technique called Multiversion Concurrency Control, or MVCC in short.