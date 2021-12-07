# Some Practice Tips

* `std::endl` must be appened to `std::cout`

The reason for this is that typically endl flushes the contents of the stream.

You must implement at least one `std::endl` before exit of a program.

* `std::queue`, `std::deque` and `std::stack`

`deque`: Double ended queue, insert and remove from both ends

`queue`: insert only in one end and remove from the other (first in first out)

`stack`: LIFO context (last-in first-out)