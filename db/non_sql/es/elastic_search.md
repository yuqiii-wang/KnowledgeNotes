# Elastic Search

*Elastic Search* is characterized by its

* non-sql json-alike document storage
* provided rich indexing and search methods, e.g., search bm25

P.S. below are for ES 8.13 version.

## Nodes and Roles

The below three node roles are the mostly used in elastic search.
One node can have multiple node roles.

A node is an elasticsearch instance/process.
One bare-metal machine can run multiple nodes (should config diff transport and publish ports).

Reference: https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-node.html

### Master Node

A node that has the master role (Master-eligible node), which makes it eligible to be elected as the master node.

A master node is responsible for:

* Bootstrapping a cluster

* *discovery*: the process where nodes find each other when the master is unknown, such as when a node has just started up or when the previous master has failed.

* Publishing the cluster state

### Data Node

A data node can perform CRUD, search, and aggregations.

### Ingest Node

Ingest nodes are able to apply an ingest pipeline, where a document undergoes a series of transforms before getting indexed (e.g., provided additional metadata), transforms include `lowerCase()`, `computeIDF()` (Inverted Document Frequency), etc.

## Kibana

Kibana is a UI client for ES.

## Data Sync

### Node Communications

* `network.bind_host` vs `network.publish_host`

`publish_host` means: "Call me on this number".

`bind_host` means: "I'll answer on this number".

* `network.host`: Sets the address of this node for both HTTP and transport traffic.

`network.host` will set `network.bind_host` and `network.publish_host` to the same value.

* `http.port` vs `transport.port`

`transport.port` is used for internal communication between nodes in an Elasticsearch cluster (default to 9300-9400).

`http.port` is used for HTTP client communication (defaults to 9200-9300).

## Shard and Index

### Shards

Shards are physical binary data represents documents.
Shards are immutable that when update/delete happens on a document stored in a shard, this shard is marked "DELETE" rather than physically removed from disk.
This immutability has benefits such as no worries of multi-process access.

#### Shard Persist on Disk

Once changes have been made into shards, e.g., CRUD operations on documents, such updates are first maintained in *filesystem cache*, then by `fsync` (unix system call to flush memory data to disk) to persist on disk.
New changes loaded to filesystem cache are very fast, and such changes are searchable; if wait till all changes getting flushed to disk then make the changes visible by search, would deteriorate performance by a lot.

ES maintains a *translog* to record ES CRUD operations.

### Index

An index is just a logical namespace that points to one or more physical shards.
ES uses *inverted index* (documents' token uniqueness computed as indexing, more distinct tokens a document contains, more easily this document can be found).

#### Lucene

Lucene is the backend text search engine of ES index, features include

* Text similarity based on *Edit Distance*
* Pluggable ranking models, including the Vector Space Model and Okapi BM25

### Shard Allocation

Below are ES schemes to manage shards to prevent disk space overflow.

* `cluster.routing.allocation.disk.watermark.low`, defaults to $85\%$, no more shards allocated to this node if disk usage is $\ge 85\%$
* `cluster.routing.allocation.disk.watermark.high`, defaults to $90\%$, shards would be reallocated to others nodes if disk usage is $\ge 90\%$
* `cluster.routing.allocation.disk.watermark.flood_stage`, defaults to $95\%$, shards are readonly or to be deleted (`index.blocks.read_only_allow_delete`), so that shards would not use more disk space.

