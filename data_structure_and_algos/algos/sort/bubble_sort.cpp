#include <vector>
#include <iostream>
#include <bits/stdc++.h>


std::vector<int>& bubbleSort(std::vector<int>& vec)
{
    for (int i = 0; i < vec.size(); i++) {
        for (int j = i; j < vec.size(); j++) {
            if (vec[i] < vec[j]) {
                std::swap(vec[i], vec[j]);
            }
        }
    }

    return vec;
}

int main()
{
    std::vector<int> vec{1,2,3,2,3,4,7,6};
    bubbleSort(vec);

    for (auto& v : vec) {
        std::cout << v << " ";
    }
    std::cout << std::endl;

    return 0;
}