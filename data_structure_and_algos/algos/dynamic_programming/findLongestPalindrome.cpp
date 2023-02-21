#include <iostream>
#include <vector>
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
        // dp[i][j] represents if s[i..j] is a palindrome
        vector<vector<int>> dp(n, vector<int>(n));
        // init: all chars itself form palindromes
        for (int i = 0; i < n; i++) {
            dp[i][i] = 1;
        }
        
        // substring length iteration, a palindrome should at least has two chars
        for (int L = 2; L <= n; L++) {
            // left boundary of the substring
            for (int i = 0; i < n; i++) {
                // right boundary of the substring
                int j = L + i - 1;
                // if the right boundary breaches total string size, break the loop
                if (j >= n) {
                    break;
                }

                if (s[i] != s[j]) {
                    dp[i][j] = 0;
                } else {
                    if (j - i < 3) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = dp[i + 1][j - 1];
                    }
                }

                // if dp[i][L] == true, then s[i..L] is palindrome, and find the max lengths
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