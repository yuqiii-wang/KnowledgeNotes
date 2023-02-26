#include <iostream>
#include <vector>
#include <cstring>

/*
    Given a string s, return the longest palindromic substring in s.
    
    Example 1:

    Input: s = "babad"
    Output: "bab"
    Explanation: "aba" is also a valid answer.
*/

using namespace std;

class SolutionByDynamicProgramming {
public:
    static string longestPalindrome(string s) {

        int n = s.size();

        if (n < 2) {
            return s;
        }
        if (n == 2 && s[0] == s[1]) {
            return s;
        }

        int maxLen = 1;
        int begin = 0;

        bool dp[n][n];
        for (int k = 0; k < n; k++) {
            std::memset(dp[k], false, n*sizeof(bool));
            dp[k][k] = true;
        }

        // L for length
        for (int L = 2; L <= n; L++) {
            for (int i = 0; i < n; i++) {

                int j = i + L - 1;
                if (j >= n) break;

                if (s[i] != s[j]) {
                    dp[i][j] = false;
                }
                else {
                    if (j - i < 3) {
                        dp[i][j] = true;
                    } else {
                        dp[i][j] = dp[i+1][j-1]; // critical, 
                                                 // it remembers the last state true/false from where it proceeds
                    }
                }

                if (dp[i][j] && j - i + 1 > maxLen) {
                    maxLen = j - i + 1;
                    begin = i;
                }
            }
        }

        return s.substr(begin, maxLen);
    }
};

int main()
{
    string a{"abcd"}; // a
    string b{"abcdcba"}; // abcdcba
    string c{"abcdca"}; // cdc
    string d{"aabbaa"}; // aabbaa
    string e{"aababac"}; // ababa
    string f{"babbbbbcb"}; // bbbbb

    cout << SolutionByDynamicProgramming::longestPalindrome(a) << endl;
    cout << SolutionByDynamicProgramming::longestPalindrome(b) << endl;
    cout << SolutionByDynamicProgramming::longestPalindrome(c) << endl;
    cout << SolutionByDynamicProgramming::longestPalindrome(d) << endl;
    cout << SolutionByDynamicProgramming::longestPalindrome(e) << endl;
    cout << SolutionByDynamicProgramming::longestPalindrome(f) << endl;

    return 0;
}