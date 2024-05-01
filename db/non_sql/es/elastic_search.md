# Elastic Search

*Elastic Search* is characterized by its

* non-sql json-alike document storage
* provided rich indexing and search methods, e.g., search bm25

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

## Data Sync

* Transport port

Used for internal communication between nodes in an Elasticsearch cluster (default to 9300).
