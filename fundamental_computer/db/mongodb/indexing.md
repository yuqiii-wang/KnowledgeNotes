# MongoDB Indexing

Indexes are applied at the collection level, and can store the value of a specific field or set of fields, ordered by the value of the field. MongoDB indexes use a B-tree data structure.

When MongoDB imports data into a collection, it will create a primary key `_id` that is enforced by an index, and is a reserved filed name. 

While creating index, MongoDB server sort of processes all the documents in the collection and create an index sheet, this would be time consuming understandably if there are too many documents; as well as creating when documents are inserted.

MongoDB uses ascending (1) or descending (-1) as sort order.

* Always use index when querying/sorting docs

## Index Types

* Unique Index

Unique index should be unique to each doc, declared by
```js
db.users.createIndex({username: 1}, {unique: true})
```

Declare drop duplicate keys to avoid multiple docs having same unique key err:
```js
db.users.createIndex({username: 1}, {unique: true, dropDups: true})
```

MongoDB by default has a field `_id`: a unique index because it’s the primary key.

* Sparse Index

It refers to docs without a declared index field, and they can be queried by
```js
db.products.find({category_id: null})
```
or declared by
```js
db.products.createIndex({country: 1}, {unique: true, sparse: true})
```

* Multi-key Index

* Hash Key

Hash Key allows fast access for `$eq` operation.

```js
db.products.createIndex({product_name: 'hashed'})
```

* Covered Index

An index can be said to cover a
query if all the data required by the query resides in the index itself.
 
Covered index queries are also known as index-only queries because these queries are served without having to reference the indexed documents themselves.