# MongoDB

MongoDB: json format document storage.

## Index

Indexes are applied at the collection level, and can store the value of a specific field or set of fields, ordered by the value of the field. MongoDB indexes use a B-tree data structure.

When MongoDB imports data into a collection, it will create a primary key `_id` that is enforced by an index, and is a reserved filed name. 

While creating index, MongoDB server sort of processes all the documents in the collection and create an index sheet, this would be time consuming understandably if there are too many documents; as well as creating when documents are inserted.

MongoDB uses ascending (1) or descending (-1) as sort order.

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
```bash
# syntax
db.collection.find(query, projection)

# example, search for documents with qty equal to 10 and price great than 100
use db_name
db.collection_name.find( {qty: {$eq: 10}, price: {$gt: 100}} )
```

Delete
```bash
# syntax
db.collection.deleteMany(query)

# example, delete all
db.collection_name.deleteMany({})
# example, delete by condition (qty greater than 10)
db.collection_name.deleteMany({qty: {$gt: 10}})
```