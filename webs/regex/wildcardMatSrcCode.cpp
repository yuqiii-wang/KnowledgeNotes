// Wildcard Matching Source Code for * and ?
// https://leetcode.com/problems/wildcard-matching/

/****
 Given an input string (s) and a pattern (p), implement wildcard pattern matching with support for '?' and '*'.
 
 '?' Matches any single character.
 '*' Matches any sequence of characters (including the empty sequence).
 
 Example:
 Input:
 s = "adceb"
 p = "*a*b"
 Output: true
 Explanation: The first '*' matches the empty sequence, while the second '*' matches the substring "dce".
 ****/

#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    Solution(){};
    bool isMatch(string s, string p);
};

bool Solution::isMatch(string s, string p){
    if (s.empty() && p.empty())
        return true;
    
    // special cases when str is empty but pattern is not
    if (s.empty() && p.size() > 0){
        if ((p.size() == 1)){
            if (p[0] == '*'){
                return true;
            }
            else{
                return false;
            }
        }
        
        bool isOnlyStar = true;
        for (int i = 1; i < p.size(); i++){
            isOnlyStar &= hash<char>{}(p[i]) == hash<char>{}(p[i-1]);
        }
        return isOnlyStar;
    }
    
    // init
    int str_size = static_cast<int>(s.size());
    int pat_size = static_cast<int>(p.size());
    vector<bool> vec(pat_size+1, false);
    vector<vector<bool>> dp(str_size+1, vec); // dynamic programming
    
    // base case, that when starting with a star, dp should be true
    dp[0][0] = true;
    for (int i = 1; i <= pat_size; i++){
        if (p[i-1] == '*')
            dp[0][i] = dp[0][i-1];
        else
            break;
    }
    
    // dynamic programming
    for (int i = 1; i <= str_size; i++){
        for (int j = 1; j <= pat_size; j++){
            if ((s[i-1] == p[j-1] || p[j-1] == '?') && dp[i-1][j-1])
                dp[i][j] = true;
            else if (p[j-1] == '*' && (dp[i-1][j] || dp[i][j-1]))
                dp[i][j] = true;
        }
    }
    
    return dp[str_size][pat_size];
}

int main(){
    Solution* S = new Solution();
    
    string s1 = {"aabb"};
    string p1 = {"*a*b"};
    cout << S->isMatch(s1, p1) << endl;

    string s2 = {""};
    string p2 = {"*"};
    cout << S->isMatch(s2, p2) << endl;
    
    string s3 = {"aa"};
    string p3 = {"a"};
    cout << S->isMatch(s3, p3) << endl;
    
    string s4 = {""};
    string p4 = {"*a"};
    cout << S->isMatch(s4, p4) << endl;
    
    return 0;
}
