# SQLite

SQLite provides minimal SQL services. In regards to concurrency, there are

* Read is in safe concurrency

First, by default, multiple processes can have the same SQLite database open at the same time, and several read accesses can be satisfied in parallel.

* Write implements a db-level lock

In case of writing, a single write to the database locks the database for a short time, nothing, even reading, can access the database file at all.

* Enabling "Write Ahead Logging (WAL)" for read/write concurrency

Beginning with version 3.7.0, a new "Write Ahead Logging" (WAL) option is available, in which reading and writing can proceed concurrently.

By default, WAL is not enabled. To turn WAL on, refer to the SQLite documentation.