// find the number of paths that a robot can move from top-left corner to the bottom-right corner of a rectangle/matrix

// dynamic programming: a cell's number of paths is the sum of previous cells' number of paths f(i,j) = f(i-1, j) + f(i, j-1)

class Solution {
public:
    int uniquePaths(int m, int n) {
        if (m == 1 && n == 1) {
            return 0;
        }
        else if ((m == 1 && n == 2) || (m == 2 && n == 1)) {
            return 1;
        }

        int dp[m][n];
        dp[0][0] = 0;

        for (int i = 1; i < m; i++) {
            dp[i][0] = 1;
        }
        for (int i = 1; i < n; i++) {
            dp[0][i] = 1;
        }

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }
};