#include <unordered_map>
#include <iostream>
#include <vector>

/*
    Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

    You may assume that each input would have **exactly one solution**, and you may not use the same element twice.

    Input: nums = [2,7,11,15], target = 9
    Output: [0,1]
    Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].
*/

using namespace std;

class SolutionBruteForce {
public:
    static vector<int> twoSum(vector<int>& nums, int target) {
        for (int i = 0; i < nums.size(); i++) {
            for (int j = 0; j < nums.size(); j++) {
                // exactly one solution and no self sum, hence i != j
                if (i != j && nums[i] + nums[j] == target) {
                    return vector<int>{i,j};
                }
            }
        }
        return vector<int>();
    }
};

class SolutionHashTable {
public:
    static vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> hashTable;
        for (int i = 0; i < nums.size(); i++) {
            auto it = hashTable.find(target - nums[i]);
            if (it != hashTable.end()) // iterator != end() indicates found item
                return vector<int>{i, it->second};
            hashTable[nums[i]] = i;
        }
        return vector<int>();
    }
};
