// find all Permutations of a vector of numbers
// Input: nums = [1,2,3]
// Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

#include <vector>
#include <unordered_set>
#include <iostream>

using namespace std;

struct hashVev {
    long operator()(std::vector<int> const& vec) const {
        std::hash<uint32_t> h;
        long ret = vec.size();
        for(auto& i : vec) {
            ret ^= h(i) | i;
        }
        return ret;
    }
};

class Solution {
private:
    vector<int> visited;
    void dfs(vector<int>& nums, vector<int>& tmp_nums, unordered_set<vector<int>, hashVev>& ans, int depth, int len) {
        if (tmp_nums.size() == len) {
            ans.insert(tmp_nums);
            return;
        }

        for (int i = depth; i < len; i++) {
            swap(nums[i], nums[depth]);
            tmp_nums.push_back(nums[depth]);
            dfs(nums, tmp_nums, ans, depth+1, nums.size());
            tmp_nums.pop_back();
            swap(nums[i], nums[depth]);
        }

        return;
    }

public:
    vector<vector<int>>permuteUnique(vector<int>& nums) {
        unordered_set<vector<int>, hashVev> ansSet;
        vector<int> tmp_nums;
        visited.resize(nums.size());
        for (auto& elem : visited) {
            elem = 0;
        }
        dfs(nums, tmp_nums, ansSet, 0, nums.size()); 

        vector<vector<int>> ans;
        for (auto& elem : ansSet) {
            ans.push_back(elem);
        }

        return ans;
    }
};

int main()
{
    Solution sol;
    vector<int> vec{1,1,2,2};

    vector<vector<int>> ans = sol.permuteUnique(vec);
    for (auto& elem : ans) {
        cout << "[" ;
        for (auto& num : elem) {
            cout << num << ",";
        }
        cout << "] " ;
    }
    cout << endl;

    return 0;
}