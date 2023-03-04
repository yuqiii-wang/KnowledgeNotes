#include <vector>
#include <iostream>
#include <algorithm>

/*
    Delete odd idx elements from an array:
    Input: 1,2,3,4,5,6,6,5,4
    Output: 1,3,5,6,4

    should use `erase_if` (since C++20)
*/

int main()
{
    std::vector<int> v{1,2,3,4,5};
    std::erase_if(v, [&v](auto& e){
        return (&e - &v[0] + 1) % 2 == 0;
    });
    for(auto& i: v) {
        std::cout<<i<<", ";
    }

    return 0;
}