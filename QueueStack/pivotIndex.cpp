// Given an array of integers nums, write a method that returns the "pivot" index of this array.

// We define the pivot index as the index where the sum of the numbers to the left of the index is equal to the sum of the numbers to the right of the index.

// If no such index exists, we should return -1. If there are multiple pivot indexes, you should return the left-most pivot index.
#include <vector>
#include <numeric>
#include <iostream>

#ifndef NULL
#define NULL nullptr // -std=c++11
#endif

using namespace std;

class Solution {
public:
    int pivotIndex(vector<int>& nums) {
        long int sizeNum = nums.size();

        for (size_t i = 0; i < sizeNum; ++i){
            long int left = accumulate(&nums[0], &nums[i], 0);
            long int right = accumulate(&nums[i+1], &nums[sizeNum], 0);
            if (left == right) return i;
        }
        return -1;
    }
};

int main(){
    int arr[] = { 10, 20, 30, 40, 50 }; 
    int n = sizeof(arr) / sizeof(arr[0]); 
    vector<int> num(arr, arr + n); 

    Solution* SolutionObj = new Solution();
    SolutionObj->pivotIndex(num);
    return 0;
}