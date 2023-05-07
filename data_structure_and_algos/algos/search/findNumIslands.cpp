// find the number of islands (`1` for land and `0` for ocean)
// Input: grid = [
//   ["1","1","1","1","0"],
//   ["1","1","0","1","0"],
//   ["1","1","0","0","0"],
//   ["0","0","0","0","1"]
// ]
// Output: 2

// By DFS, set already visited cells to `2`

#include <vector>

using namespace std;

class Solution {
private:
    void dfs(vector<vector<char>>& grid, int r, int c) {
        if (!inArea(grid, r, c)) return;
        if (grid[r][c] != '1') return;

        grid[r][c] = '2'; // marked as visited

        dfs(grid, r-1, c);
        dfs(grid, r+1, c);
        dfs(grid, r, c-1);
        dfs(grid, r, c+1);
    }

    bool inArea(vector<vector<char>>& grid, int r, int c) {
        return r >= 0 && r < grid.size()
            && c >= 0 && c < grid[0].size();
    }
public:
    int numIslands(vector<vector<char>>& grid) {
        int cntIslands = 0;
        for (int i = 0; i < grid.size(); i++) {
            for (int j = 0; j < grid[0].size(); j++) {
                if (grid[i][j] == '1') {
                    cntIslands++;
                    dfs(grid, i, j);
                }
            }
        }
        return cntIslands;
    }
};