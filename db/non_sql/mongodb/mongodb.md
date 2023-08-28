# MongoDB

MongoDB: json format document storage.

* `BSON`

`BSON` is based on the term JSON and stands for "Binary JSON", such as `{"hello": "world"}` be in the below formatting
```cpp
\x16\x00\x00\x00          // total document size
\x02                      // 0x02 = type String
hello\x00                 // field name
\x06\x00\x00\x00world\x00 // field value (size of value, value, null terminator)
\x00                      // 0x00 = type EOO ('end of object')
```

* Shard

Shards (when enabled) sit between collection and document: a group of documents clustered by values on a field.

## MongoDB Shell Cmds

Indxing
```bash
# syntax
db.collection.createIndex( <key and index type specification>, <options> )

# example
use db_name
db.collection_name.createIndex(
  { item: 1, quantity: -1 } ,
  { name: "item_query_idx" }
)
```

Query, plese refer to this https://docs.mongodb.com/manual/reference/method/db.collection.find/ to see detail, including query operators with prefix `$`
```js
// syntax
db.collection.find(query, projection)

// example, search for documents with qty equal to 10 and price great than 100
use db_name
db.collection_name.find( {qty: {$eq: 10}, price: {$gt: 100}} )
```

Delete
```js
// syntax
db.collection.deleteMany(query)

// example, delete all
db.collection_name.deleteMany({})
// example, delete by condition (qty greater than 10)
db.collection_name.deleteMany({qty: {$gt: 10}})
```

Existing field search:

When the qty field exists and its value does not equal 5 or 15.
```js
db.collection_name.find({"qty": {"$exists": true, "$nin": [5,15]}})
```

Explain:

`explain` tells about query operation details such as number of docs scanned and elapse time.

```js
db.collection_name.find({}).sort({close: -1}).limit(1).explain()
```

Statistics of a collection:

`stats` tells about statistics of a collection, such as sizes, index, counts.

```js
db.collection_name.stats()
```

### Shell Scripts

* Mongos: MongoDB Shard Utility.

* Mongod: The primary daemon process for the MongoDB system. It handles data requests, manages data access, and performs background management operations.

## Mongo vs MySQL

* REPEATABLE READ consistency: MySQL supports complex transactions over multiple rows/records, while MongoDB only supports single document write/read consistency
* MongoDB has built-in sharding while MySQL does not have it
* MongoDB tries loading all collections/tables in memory (thus high consumption of memory), while Mysql is not
* MongoDB does not have `JOIN` statement at hand to joining multiple collections/tables; instead, need to prepare a view `db.createView(...)` in advance if required to search across multiple collections/tables.

## Storage Engine

Starting in MongoDB 3.2, the *WiredTiger* storage engine is the default storage engine.

* Power of 2 Sized Allocations Per Document

By default, memory allocation grows by the power of 2 per document.

For example, a document is sized 15 KB, MongoDB allocates 16 KB memory; then when this document is updated to have size of 17 KB, MongoDB allocates 32 KB memory (re-allocation similar to that of c++ vector).

* Snapshots and Checkpoints

WiredTiger uses MultiVersion Concurrency Control (MVCC).
A snapshot presents a consistent view of the in-memory data.

Starting in version 3.6, MongoDB configures WiredTiger to create checkpoints (i.e. write the snapshot data to disk) at intervals of 60 seconds.

* Journal

WiredTiger uses a write-ahead log (i.e. journal) that persists all data modifications between checkpoints.

Journal's records are synced to disk every 100 ms.
Journal's records are checked against snapshots/checkpoints every 60 secs.

## Capped Collection

*Capped Collection* are fixed-size, similar to circular buffers: once a collection fills its allocated space, it makes room for new documents by overwriting the oldest documents in the collection.

Setup by
```js
db.createCollection( "log", { capped: true, size: 100000 } )
```

Capped collections follow FIFO order that the most recent document data is stored.

<div style="display: flex; justify-content: center;">
      <img src="imgs/mongodb_capped_coll_fifo.png" width="30%" height="15%" alt="mongodb_capped_coll_fifo" />
</div>
</br>

### Use Scenarios

Capped collection is fixed-sized, hence it can only store a limited number of documents.

It can be used a cache for hot data storage (Redis-like services).

Usually, new document data is inserted being appended (typically indexed by timestamp) to the end of the FIFO.
There should be few update/delete actions for such actions can cause B-tree restructure.
Old document data will be auto removed anyway.