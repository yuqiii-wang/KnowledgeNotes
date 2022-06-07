//https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/

//Select stocks that make max profit.
//Input: [3,2,6,5,0,3], k = 2
//Output: 7
//Explanation: Buy on day 2 (price = 2) and sell on day 3 (price = 6), profit = 6-2 = 4.
//Then buy on day 5 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.

#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    int maxProfit(int k, vector<int>& prices);
    Solution(){};
};

int Solution::maxProfit(int k, vector<int>& prices){
    if (prices.size() <= 1)
        return 0;
    
    if (static_cast<int>(prices.size() / 2) <= k){ // when there are more transactions than required
        int maxProf = 0;
        for (int i = 1; i < prices.size(); i++){
            if (prices[i] > prices[i-1]){
                maxProf += prices[i] - prices[i-1];
            }
        }
        return maxProf;
    }
    
    vector<int> vec(prices.size(), 0);
    vector<vector<int>> dp(k+1, vec); // dynamic programming
    int tmpMax;
    
    // example:
    // given an array [3,2,6,5,0,3] with 2 transactions, dp will produce the following results:
    // [0, 0, 0, 0, 0, 0]
    // [0, 0, 4, 4, 4, 4]
    // [0, 0, 4, 4, 4, 7]
    for (int i = 1; i < k+1; i++){
        tmpMax = dp[i-1][0] - prices[0];    // the first elem should be subtracted as
                                            // the next elem's price will cover it
        for (int j = 1; j < prices.size(); j++){
            tmpMax = max(tmpMax, dp[i-1][j-1] - prices[j-1]);
            dp[i][j] = max(dp[i][j-1], tmpMax + prices[j]);
        }
    }
    return dp[k][prices.size()-1];
}

int main(){
    Solution* S = new Solution();
    
    vector<int> arr_1 = {1,2,3,4,3,4,1,5};
    cout << S->maxProfit(2, arr_1) << endl;
    
    vector<int> arr_2 = {3,2,6,5,0,3};
    cout << S->maxProfit(2, arr_2) << endl;
    
    vector<int> arr_3 = {2,4,1};
    cout << S->maxProfit(2, arr_3) << endl;
    
    vector<int> arr_4 = {1,2,4,2,5,7,2,4,9,0};
    cout << S->maxProfit(4, arr_4) << endl;
    
    return 0;
}
