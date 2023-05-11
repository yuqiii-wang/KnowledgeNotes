// generate all valid parentheses given a number

// input: n = 3
// output: ["((()))","(()())","(())()","()(())","()()()"]



#include <vector>
#include <iostream>

using namespace std;

class Solution {
private:
    void dfs(vector<string>& ans, string& tmp, int depth, int len) {
        if (2 * len == depth) {
            ans.push_back(tmp);
            return ;
        }

        
    }
public:
    vector<string> generateParenthesis(int n) {
        vector<string> ans;

        return ans;
    }
};

int main()
{
    Solution sol;

    int num = 3; //num of parenthesis pair

    vector<string> ans = sol.generateParenthesis(num);
    for (auto& elem : ans) {
        for (auto& num : elem) {
            cout << num << ",";
        }
    }
    cout << endl;

    return 0;
}