#include <vector>
#include <iostream>

void insertion_sort(std::vector<int>& vec)
{
    for (int i = 1; i < vec.size(); i++) {
        int key = vec[i];
        int j = i - 1;
        while (j >= 0 && vec[j] > key) {
            std::swap(vec[j+1], vec[j]);
            j--;
        }
    }
}

int main()
{
    std::vector<int> vec{1,2,3,2,3,4,7,6};
    insertion_sort(vec);

    for (auto& v : vec) {
        std::cout << v << " ";
    }
    std::cout << std::endl;

    return 0;
}