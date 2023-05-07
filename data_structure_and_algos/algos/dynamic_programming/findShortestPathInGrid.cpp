// dynamic programming: f_min[i][j] = min(f_min[i-1][j], f_min[i][j-1]) + f[i][j]

#include <vector>
#include <iostream>

using namespace std;

class Solution {
public:
    static int minPathSum(vector<vector<int>>& grid) {
        if (grid.size() == 1 && grid[0].size() == 1) {
            return grid[0][0];
        }

        int gridSum[grid.size()][grid[0].size()];
        std::pair<int, int> gridSumPair[grid.size()][grid[0].size()];
        gridSum[0][0] = grid[0][0];
        gridSumPair[0][0] = {0, 0};

        for (int i = 1; i < grid.size(); i++) {
            gridSum[i][0] = gridSum[i-1][0] + grid[i][0];
            gridSumPair[i][0] = {i-1, 0};
        }
        for (int i = 1; i < grid[0].size(); i++) {
            gridSum[0][i] = gridSum[0][i-1] + grid[0][i];
            gridSumPair[0][i] = {0, i-1};
        }

        for (int i = 1; i < grid.size(); i++) {
            for (int j = 1; j < grid[0].size(); j++) {
                if (gridSum[i-1][j] < gridSum[i][j-1]) {
                    gridSum[i][j] = gridSum[i-1][j] + grid[i][j];
                    gridSumPair[i][j] = {i-1, j};
                }
                else {
                    gridSum[i][j] = gridSum[i][j-1] + grid[i][j];
                    gridSumPair[i][j] = {i, j-1};
                }
            }
        }

        std::vector<std::pair<int, int>> path;
        path.reserve(grid.size()*2);
        path.push_back({grid.size()-1,grid[0].size()-1});
        for (int i = grid.size()-1, j = grid[0].size()-1, i_tmp = i, j_tmp = j; i > 0 || j > 0; ) {
            i_tmp = gridSumPair[i][j].first;
            j_tmp = gridSumPair[i][j].second;
            path.push_back({i_tmp,j_tmp});
            i = i_tmp;
            j = j_tmp;
        }

        for (auto& pr : path) {
            std::cout << "[" << pr.first << "," << pr.second << "] ";
        }
        std::cout << std::endl;

        return gridSum[grid.size()-1][grid[0].size()-1];
    }
};

int main()
{
    vector<vector<int>> grid{{-1,0,0,0}, {0,-1,0,0}, {0,0,-1,1}, {0,0,0,0}};
    int bonuses = std::abs(Solution::minPathSum(grid));
    std::cout << "bonuses: " << bonuses << std::endl;

    return 0;
}