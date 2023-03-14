# Cache

Cache stores recently/frequently used data so that next time it can fast provide the same data.

## Cache entries

Data is transferred between memory and cache in blocks of fixed size, called cache lines or cache blocks.
When a cache line is copied from memory into the cache, a cache entry is created. 
The cache entry will include the copied data as well as the requested memory location (called a tag).

When the processor needs to read or write a location in memory, it first checks for a corresponding entry in the cache. 

A `CACHE_LINE_SIZE` is 64 bytes, typically. 
So that cache prefetches 64 bytes of data every time.
For example, `CACHE_LINE_SIZE/sizeof(int)` is 16, and the below `a[32]` should use two cache entries.
```cpp
int a[32];
for (int i = 0; i < 32; ++i) {
    a[i]++;
}
```

### Cache-Friendly Code

* Keep most of our data in contiguous memory locations.

In a loop, access data by row not by column.

Prefer array and vector over list or hash table for contiguous memory access.

* Keep function stack small

Typically, avoid using recursion .

* Avoiding Out-of-Order Jumps

Conditional jumps can cause slowdowns because of branching, that code execution not linear and CPU cannot prefetch instructions and data.

## Cache Scheduling Policies

### Simple queue-based policies - First in first out (FIFO)

Using this algorithm the cache behaves in the same way as a FIFO queue. 
The cache evicts the blocks in the order they were added, without any regard to how often or how many times they were accessed before.

### Simple recency-based policies - Least recently used (LRU)

Discards the least recently used items first; 
keep track of what was used when, which is expensive if one wants to make sure the algorithm always discards the least recently used item.

Assume a cache can hold $n$ data items. The cache records the numbers of hit for all data item. The least recently used data item.

* Tutorial implementation:

`LRUCache(int capacity)` Initialize the LRU cache with positive size `capacity`.

`int get(int key)` Return the value of the `key` if the key exists, otherwise return `-1`.

`void put(int key, int value)` Update the value of the `key` if the key exists. Otherwise, add the `key-value` pair to the cache. If the number of keys exceeds the `capacity` from this operation, evict the least recently used key.

The functions get and put must each run in $O(1)$ average time complexity.

Example Usage
```cpp
LRUCache* lRUCache = new LRUCache(2);
lRUCache->put(1, 1); // cache is {1=1}
lRUCache->put(2, 2); // cache is {1=1, 2=2}
lRUCache->get(1);    // return 1
lRUCache->put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
lRUCache->get(2);    // returns -1 (not found)
```

Implementation
```cpp

class LRUCache {
  int capacity;
  list<int> lru_list;
  unordered_map<int, pair<int, list<int>::iterator>> cache;
  
public:
  LRUCache(int capacity) : capacity(capacity) {}
  
  int get(int key) {
    auto it = cache.find(key);
    if (it == cache.end()) return -1;
    
    lru_list.erase(it->second.second);
    lru_list.push_front(key);
    it->second.second = lru_list.begin();
    return it->second.first;
  }
  
  void put(int key, int value) {
    auto it = cache.find(key);
    if (it != cache.end()) {
      lru_list.erase(it->second.second);
    }
    lru_list.push_front(key);
    cache[key] = {value, lru_list.begin()};
    if (cache.size() > capacity) {
      int last = lru_list.back();
      lru_list.pop_back();
      cache.erase(last);
    }
  }
};
```

* Boost Implementation

## Cache Hit Measurement by Google Benchmark