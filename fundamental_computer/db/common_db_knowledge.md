# Common DB knowledge

## Database storage structures

DB Storage with indexing:
ordered/unordered flat files, ISAM, heap files, hash buckets, or B+ trees

**Unordered** storage offers good insertion efficiency ( $O(1)$ ), but inefficient retrieval times ( $O(n)$ ). Most often, by using indexes on primary keys, retrieval times of $O(log n)$ or $O(1)$ for keys are the same as the database row offsets within the storage system.

**Ordered** storage typically stores the records in order. It has lower insertion efficiency, while providing more efficient retrieval of $O(log n)$

### **Structured files**

* Heap files

Heap files are lists of unordered records of variable size. New records are added at the end of the file, providing chronological order.

* Hash buckets

Hash functions calculate the address of the page in which the record is to be stored based on one or more fields in the record. So that given fields of a record (served as indexing), hash function can point to the memory address with $O(1)$.

* B+ trees

A B+ tree is an m-ary tree with a variable but often large number of children per node. B+ trees have very high fanout (number of pointers to child nodes in a node, typically on the order of 100 or more).

It searches with multiple branches, thus efficient
```py
def search(k):
    return tree_search(k, root)

def tree_search(k, node):
    if node is a_leaf:
        return node
    switch (k):
        case k ≤ k_0
            return tree_search(k, p_0)
        case k_i < k ≤ k_{i+1}
            return tree_search(k, p_{i+1})
        case k_d < k
            return tree_search(k, p_{d})
```

### **Data Orientation**

* "row-oriented" storage: 

each record/entry as a unit

* "column-oriented" storage: 

feature based storage

easy for data warehouse-style queries