// https://leetcode.com/explore/learn/card/queue-stack/231/practical-application-queue/1374/
// g++ -std=c++11 findIslands.cpp

#include <vector>

using namespace std;

class Solution {
public:
    int numIslands(vector<vector<char>>& grid) {
    int ctr = 0;
     for (int i = 0; i < grid.size(); i++){
         for (int j = 0; j < grid[0].size(); j++){
             ctr += findIsland(grid, i, j);
         }
     }
    return ctr;
    }
private:
    int findIsland(vector<vector<char>>& grid, int x, int y){
        if (x < 0 || x > grid.size()-1 || y < 0 || y > grid[0].size()-1 || grid[x][y] == '0')
            return 0;
        grid[x][y] = '0';
        findIsland(grid, x-1, y);
        findIsland(grid, x+1, y);
        findIsland(grid, x, y-1);
        findIsland(grid, x, y+1);
        return 1;
    }
};

int main(void){
    Solution();
    return 0;
}