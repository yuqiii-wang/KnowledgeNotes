// find all Permutations of a vector of numbers
// Input: nums = [1,2,3]
// Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

#include <vector>
#include <iostream>

using namespace std;

class Solution {
private:
    void dfs(vector<int>& nums, vector<int>& tmp_nums, vector<vector<int>>& ans, int depth, int len) {
        if (tmp_nums.size() == len) {
            ans.push_back(tmp_nums);
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
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> ans;
        vector<int> tmp_nums;
        dfs(nums, tmp_nums, ans, 0, nums.size());

        return ans;
    }
};

int main()
{
    Solution sol;
    vector<int> vec{1,2,3};

    vector<vector<int>> ans = sol.permute(vec);
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