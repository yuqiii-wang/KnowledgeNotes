#include <iostream>
#include <unordered_set>
/*
Given a string s, find the length of the longest substring without repeating characters. 

    Example 1:

    Input: s = "abcabcbb"
    Output: 3
    Explanation: The answer is "abc", with the length of 3.

*/

class SolutionBruteForce {
public:
    int lengthOfLongestSubstring(std::string s) {
        int max = 0;
        for (int i = 0; i < s.size(); i++)
        {
            int maxTemp = 0;
            std::unordered_set<char> set;
            for (int j = i; j < s.size(); j++)
            {
                if (set.find(s[j]) != set.end())
                {
                    break;
                }
                else 
                {
                    set.emplace(s[j]);
                    maxTemp++;
                }
            }
            if (maxTemp > max)
                max = maxTemp;
        }
        return max;
    }
};

/*
    Similar approach but with the improvements:
    no repeated hash set construction/destruction by `occ.erase(s[i - 1]);`
    used a left pointer and a right pointer (between which is the unique char substring),
    the second loop starts at rk+1;
*/
class SolutionSkipReindexing {
public:
    int lengthOfLongestSubstring(std::string s) {
        // hash set to record if a char has been present before
        std::unordered_set<char> occ;
        int n = s.size();
        // rk: right moving pointer, init to -1 representing not yet started moving in `s`
        int rk = -1, ans = 0;
        // i: left moving pointer
        for (int i = 0; i < n; ++i) {
            if (i != 0) {
                // erase the previous char at a time
                occ.erase(s[i - 1]);
            }
            while (rk + 1 < n && !occ.count(s[rk + 1])) {
                // keep moving rk to the right
                occ.insert(s[rk + 1]);
                ++rk;
            }
            // find the max val
            ans = std::max(ans, rk - i + 1);
        }
        return ans;
    }
};

