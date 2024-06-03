# Elastic Search

*Elastic Search* is characterized by its

* non-sql json-alike document storage
* provided rich indexing and search methods, e.g., search bm25

P.S. below are for ES 8.13 version.

## Nodes and Roles

Node roles are defined in `node.roles` from `config/elasticsearch.yml`
One node can have multiple node roles.

A node is an elasticsearch instance/process.
One bare-metal machine can run multiple nodes (should config diff transport and publish ports).

An *eligible node* refers to a node that can hold or manage a specific type of shard, whether it's a primary shard or a replica shard.

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

### Coordinating Node

A coordinating node decides how to delegate a request to what nodes for process, e.g., document search or update.

There are two phases of a request process from a coordinating node's perspective:

1. In the scatter phase, the coordinating node forwards the request to the data nodes which hold the data. Each data node executes the request locally and returns its results to the coordinating node.
2. In the gather phase, the coordinating node reduces each data node's results into a single global result set.

Every node is implicitly a coordinating node (a node that has an explicit empty list of roles via `node.roles` will only act as a coordinating node).

## Data Communication Between Nodes

### Node Communications

* `network.bind_host` vs `network.publish_host`

`publish_host` means: "Call me on this number".

`bind_host` means: "I'll answer on this number".

* `network.host`: Sets the address of this node for both HTTP and transport traffic.

`network.host` will set `network.bind_host` and `network.publish_host` to the same value.

* `http.port` vs `transport.port`

`transport.port` is used for internal communication between nodes in an Elasticsearch cluster (default to 9300-9400).

`http.port` is used for HTTP client communication (defaults to 9200-9300).

### How ES nodes bootstrap a cluster and vote

1. Unicast Hosts: Nodes use the `discovery.seed_hosts` to list other nodes they can initially contact.
2. When starting a new cluster, the `cluster.initial_master_nodes` setting is used to define the set of master-eligible nodes that will participate in the initial master election.
3. Nodes use the seed hosts to discover each other and communicate to form a *quorum*.
4. Master-eligible nodes vote to elect an actual managing master node
5. This managing master node is responsible for management tasks, e.g., new nodes joining, index creation, and publishing cluster states

A quorum is the minimum number of master-eligible nodes that must agree on a decision to ensure that the cluster can maintain a consistent and reliable state.
The quorum size is $(N/2) + 1$ given $N$ master-eligible nodes.

Once a node receives a quorum of votes, it becomes the master node and begins managing the cluster state. Other nodes acknowledge this node as the master.

### How a coordinating node determines what nodes to pass requests

Reference: https://www.elastic.co/guide/en/elasticsearch/reference/current/search-shard-routing.html#:~:text=By%20default%2C%20Elasticsearch%20uses%20adaptive,node%20and%20the%20eligible%20node

Search shard routing in ES is *Adaptive replica selection* (ARS) by default.
If disabled, ES routes search requests using a round-robin method.

ARS evaluates each node performance as per *Shard Allocation Awareness*, then select the best performance node.

#### Shard Allocation Awareness

ES takes into consideration a number of factors (see below) to allocate a shard

* Disk
* CPU
* RAM Memory
* Response Time
* `node.attr.server_type: hot` (optional values: `warm`, `cold`)

### How fast is a change available to read after just updated across multiple nodes

## Shard and Index

### Shards

Shards are physical binary data represents documents.
Shards are immutable that when update/delete happens on a document stored in a shard, this shard is marked "DELETE" rather than physically removed from disk.
This immutability has benefits such as no worries of multi-process access.

#### Shard Persist on Disk

Once changes have been made into shards, e.g., CRUD operations on documents, such updates are first maintained in *filesystem cache*, then by `fsync` (unix system call to flush memory data to disk) to persist on disk.
New changes loaded to filesystem cache are very fast, and such changes are searchable; if wait till all changes getting flushed to disk then make the changes visible by search, would deteriorate performance by a lot.

ES maintains a *translog* to record ES CRUD operations.

#### Shard Request Cache

Reference: https://www.elastic.co/guide/en/elasticsearch/reference/current/shard-request-cache.html

When a query request runs against an index/multiple indices, shards get executed matching against the query, and returns its local results to the coordinating node, which combines these shard-level results into a "global" result set.

Cached shards can return immediately for same queries to the ES.

A hash of the whole JSON body is used as the *cache key*, that is proof if the shard has been updated against the cached shards.
When a shard gets updated, upon refresh, the cached shard is invalidated.

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

### Primary Shards vs Replica Shards

By default, `index.number_of_replicas` is set to `1`, then upon checking node availability (or other conditions, e.g., node recovery), a replica shard is copied from a primary shard to another node.

`index.auto_expand_replicas` (Set to a dash delimited lower and upper bound (e.g. `0-5`) or use all for the upper bound (e.g. `0-all`)) allows ES to dynamically config `index.number_of_replicas`.

Usually read and write happen on primary shards, and replicas serve as a backup.

### Shard Read Write Consistency

Reference: https://www.elastic.co/guide/en/elasticsearch/reference/2.4/docs-index_.html#docs-index_

ES implements Optimistic Concurrency Control (OCC) conducting a DB transaction:

* Read Phase: A transaction reads data and performs computations, maintaining a local copy of the data it accesses.
* Validation Phase: Before committing, the transaction checks whether any of the data it read has been modified by other transactions. If there are conflicts, the transaction is aborted and retried.
* Write Phase: If validation succeeds, the transaction writes its changes to the database.

## ElasticSearch's Search

References:
https://www.elastic.co/guide/en/elasticsearch/reference/current/search-your-data.html
https://www.elastic.co/guide/en/elasticsearch/reference/current/search-search.html#search-search

ES REST API supports queries sent via below HTTP request.

* GET /\<index-name\>/_search
* GET /_search
* POST /\<index-name\>/_search
* POST /_search

ES response explained:

* `took`: elapsed time in mili-second
* all retrieved data is in `hits`
* `hits.total.value` says the number of retrieved documents
* `hits.max_score` is the highest score of documents (usually considered the most matched documents), computed by Practical Scoring Function (default to BM25)
* `hits.hits` contains all retrieved documents and associated metadata, e.g., document id
* `hits.hits._source` contains the source contents of documents

```json
{
  "took": 5,
  "timed_out": false,
  "_shards": {
    "total": 1,
    "successful": 1,
    "skipped": 0,
    "failed": 0
  },
  "hits": {
    "total": {
      "value": 1,
      "relation": "eq"
    },
    "max_score": 1.3862942,
    "hits": [
      {
        "_index": "my-index",
        "_id": "kxWFcnMByiguvud1Z8vC",
        "_score": 1.3862942,
        "_source": {
            "<source data>"
        }
      }
    ]
  }
}
```

### Multi-Index Search

Reference: https://www.elastic.co/guide/en/elasticsearch/reference/current/search-multiple-indices.html

* Search two indices `my-index-000001` and `my-index-000002`

```http
GET /my-index-000001,my-index-000002/_search
{
  "query": {
    "match": {
      "user.id": "kimchy"
    }
  }
}
```

* Search all indices with the prefix `my-index-`

```http
GET /my-index-*/_search
{
  "query": {
    "match": {
      "user.id": "kimchy"
    }
  }
}
```

* Global search

```http
GET /*/_search
{
  "query": {
    "match": {
      "user.id": "kimchy"
    }
  }
}
```

### Term Exact Search

"Term" in ES refers to exact text match (also provided regex and other flexible search methods).

For example, send the body via `POST my-index/_search?pretty` to find documents by exact match.

```json
{
  "query": {
    "term": {
      "text_field_you_want_to_search_in": "Your desired exact match words"
    }
  }
}
```

### Aggregation



### Embedding/Vector Similarity

A k-nearest neighbor (kNN) search finds the $k$ nearest vectors to a query vector, as measured by a similarity metric.

#### Approximate kNN

Approximate kNN is about searching by a distance function and would not iterate all documents.

`PUT my-text-index`

```json
{
    ...
    "my-text-vector": {
        "type": "dense_vector",
        "dims": 768,
        "similarity": "l2_norm"
      },
}
```

As of 8.13, Approximate kNN supports the below distance formula.

* `l2_norm`
* `dot_product`
* `cosine` (default option)
* `max_inner_product`

#### Exact kNN

Exact kNN uses a custom script scoring function to exhaustively visit all documents and find the match.

For example, set a score field of a doc by `PUT my-index/_doc/1`

```json
{
  "my_custom_score_field": 5
}
```

Then search this doc by a custom formula $\text{score}^2+1$ via `POST my-index/_search`

```json
{
  "script_fields": {
    "my_doubled_field": {
      "script": { 
        "source": "doc['my_custom_score_field'].value * ['my_custom_score_field'].value + 1"
      }
    }
  }
}
```

## Transaction

## Performance Config

* Shard Size

Reference: https://www.elastic.co/guide/en/elasticsearch/reference/current/size-your-shards.html

* Thread Pool

Reference: https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-threadpool.html

## Kibana

Kibana is a UI client for ES.