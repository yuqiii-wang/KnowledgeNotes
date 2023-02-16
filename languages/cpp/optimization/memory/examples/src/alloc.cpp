#include <vector>
#include <queue>
#include "MyLibAlloc.hpp"

int main()
{
    // create a vector, using MyAlloc<> as allocator
    std::vector<int,MyLib::MyAlloc<int> > v;

    // insert elements
    // - causes reallocations
    for (int i = 0; i < 10; i++)
        v.push_back(i);

    std::cout << "=======" << std::endl;

    std::deque<int,MyLib::MyAlloc<int> > q;
    for (int i = 0; i < 10; i++)
        q.push_back(i);

    return 0;
}