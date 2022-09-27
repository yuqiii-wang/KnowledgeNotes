# STL Thread Safety


## `std::unordered_map` thread safety

When multi-threading running on `std::unordered_map` performing simultaneous read and write on elements, it is NOT thread-safe.

When inserting a new element, there is rehashing, so that iterator is not valid. However, reference remains valid (element memory unchanged).