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

* Capped Collection

are fixed-size, similar to circular buffers: once a collection fills its allocated space, it makes room for new documents by overwriting the oldest documents in the collection.

Setup by
```js
db.createCollection( "log", { capped: true, size: 100000 } )
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
