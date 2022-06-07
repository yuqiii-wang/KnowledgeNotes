/*
Given an array of integers, return indices of the two numbers such that they add up to a specific target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

Example:

Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].
*/

#include <iostream> 
#include <vector> 
#include <unordered_map> 

using namespace std;

vector<int> twoSum(vector<int>& nums, int target) {
    vector<int> index;
    unordered_map<int,int> hashtable;
    for (int i = 0;i < nums.size();i++){
        hashtable[nums[i]] = i;
    }
    for (int i = 0;i < nums.size(); i ++){
        int complement = target - nums[i];
        
        // unordered_map.at() method accesses python-dict alike variables via .at(key)
        if (hashtable.count(complement) && hashtable.at(complement) != i ){
            index.push_back(i);
            index.push_back(hashtable.at(complement)); 
            return index;
            }
    }
    return index;
}

int main(void){
    vector<int> nums;
    int target;

    for (int i = 1; i <= 50; i++) 
        nums.push_back(i); 
    target = 30;

    vector<int> ret_val = twoSum(nums, target);

    cout << "Original Array\n";
    for(auto it = nums.begin(); it != nums.end(); it++) 
        cout << *it << '\t';
    cout << "\nSpecified Num:\n" << target;        
    cout << "\nFound Index:\n";
    cout << ret_val[0] << '\t' << ret_val[1] << '\n';
    return 0;
}