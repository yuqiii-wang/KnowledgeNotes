#include <vector>
#include <iostream>
#include <bits/stdc++.h>


int partition(std::vector<int>& vec, int low, int high)
{
    int i = low - 1;
    int pivot = high;
    for (int j = 0; j < high; j++)
    {
        if (vec[j] < vec[pivot]) // elems smaller than pivot are swapped to left
        {   
            i++;
            std::swap(vec[i], vec[j]);
        }
    }
    std::swap(vec[i+1], vec[high]); // i+1 is now the new pivot idx
    return i + 1;
}

void _quickSort(std::vector<int>& vec, int low, int high)
{
    // recursively calls _quickSort
    if (low < high)
    {
        int mid = partition(vec, low, high);

        _quickSort(vec, low, mid - 1);
        _quickSort(vec, mid + 1, high);
    }
}

std::vector<int>& quickSort(std::vector<int>& vec)
{
    int low = 0;
    int high = vec.size()-1;

    _quickSort(vec, low, high);

    return vec;
}

int main()
{
    std::vector<int> vec{1,2,3,2,3,4,7,6};
    quickSort(vec);

    for (auto& v : vec) {
        std::cout << v << " ";
    }
    std::cout << std::endl;

    return 0;
}