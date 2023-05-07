// from top-left corner walks to the bottom-right corner; there are obstacles in the grid
// [[0,1,0,1],
//  [0,0,0,1],
//  [1,0,0,1],
//  [0,0,0,0]]
// where `1` indicates obstacles and `0` indicates available path

#include <vector>

using namespace std;

class Solution {
public:
    static int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        if (obstacleGrid.size() == 1 && obstacleGrid[0].size() == 1) {
            if (obstacleGrid[0][0] == 1) {
                return 0;
            }
            else {
                return 1;
            }
        }
        int dp[obstacleGrid.size()][obstacleGrid[0].size()];
        for (int i = 0; i < obstacleGrid.size(); i++) {
            for (int j = 0; j < obstacleGrid[0].size(); j++) {
                dp[i][j] = 0;
            }
        }

        for (int i = 0; i < obstacleGrid.size(); i++) {
            if (obstacleGrid[i][0] == 1) {
                break;
            }
            dp[i][0] = 1;
        }
        for (int i = 0; i < obstacleGrid[0].size(); i++) {
            if (obstacleGrid[0][i] == 1) {
                break;
            }
            dp[0][i] = 1;
        }

        for (int i = 1; i < obstacleGrid.size(); i++) {
            for (int j = 1; j < obstacleGrid[0].size(); j++) {
                if (obstacleGrid[i][j] == 1) {
                    continue;
                }
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        return dp[obstacleGrid.size()-1][obstacleGrid[0].size()-1];
    }
};