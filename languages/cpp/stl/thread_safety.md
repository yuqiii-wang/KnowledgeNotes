# STL Thread Safety


### `std::unordered_map` thread safety

When multi-threading running on `std::unordered_map` performing simultaneous read and write on elements, it is NOT thread-safe.

When inserting a new element, there is rehashing, so that iterator is not valid. However, reference remains valid (element memory unchanged).

### STL Container Thread-Safe Access and Modification

STL containers are made thread safety to each element modification, such as `std::vector<Element>`, that
* simultaneous reads of the same object/element are OK
* simultaneous read/writes of different objects/elements of a container are OK

`std::vector<bool>` has each element occupying one bit of space, not necessary in contiguous space. 