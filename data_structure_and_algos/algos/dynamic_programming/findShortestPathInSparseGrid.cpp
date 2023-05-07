// dynamic programming: f_min[i][j] = min(f_min[i-1][j], f_min[i][j-1]) + f[i][j]

#include <vector>
#include <algorithm>
#include <iostream>
#include <stdlib.h> // random
#include <unordered_map>
#include <bitset>
#include <stdio.h>
#include <functional>
#include <math.h>


using namespace std;

const int gridSize = 8;

// std does not provide default pair hash for unordered_map, need to make a custom one
// A hash function used to hash a pair of any kind
struct hash_pair {
    template <typename T1, typename T2>
    long operator()(const pair<T1, T2>& p) const
    {
        auto hash1 = hash<T1>{}(p.first);
        auto hash2 = hash<T2>{}(p.second);
 
        if (hash1 != hash2) {
            return hash1 ^ hash2;             
        }
         
        // If hash1 == hash2, their XOR is zero.
        return hash1;
    }
};


class Solution {
public:

    static void fillBetweenPath(pair<int, int> startCoord, pair<int, int> endCoord, 
        unordered_map<pair<int, int>, pair<int, int>, hash_pair>& prevStepMap)
    {
        int i = endCoord.first;
        while ( i > startCoord.first ) {
            prevStepMap[{i+1,endCoord.second}] = {--i,endCoord.second};
        }
        int j = endCoord.second;
        while ( j > startCoord.second ) {
            prevStepMap[{startCoord.first,j+1}] = {startCoord.first,--j};
        }
    }

    static void maxPathSum(vector<std::bitset<gridSize>>& grid) {

        unordered_map<pair<int, int>, int, hash_pair> bonusMap;
        unordered_map<pair<int, int>, pair<int, int>, hash_pair> prevStepMap;
        prevStepMap[{0,0}] = {-1,-1};
        bonusMap[{0,0}] = 0;

        pair<int, int> startStep(0,0);
        pair<int, int> finalStep;

        for (int i = 0; i < gridSize; i++) {
            if (grid[i].none()) {
                continue;
            }
            for (int j = 0; j < gridSize; j++) {
                if ( grid[i][j] == 0x01 ) {
                    int tmpMax = 0;
                    pair<int, int> prevStep;
                    for (auto& prevBonus : bonusMap) {
                        if (prevBonus.first.second > j) { continue; }
                        if (tmpMax < prevBonus.second) {
                            tmpMax = prevBonus.second;
                            prevStep = prevBonus.first;
                        }
                    }
                    bonusMap[{i,j}] = ++tmpMax;
                    fillBetweenPath(prevStep, {i,j}, prevStepMap);

                    finalStep = {i,j}; // unordered_map does not have order, need to record the final step
                                       // the right-bottom cell bonus must have the highest score as well
                }
            }
        }

        bonusMap[{gridSize-1,gridSize-1}] = bonusMap[finalStep];
        fillBetweenPath(finalStep, {gridSize-1,gridSize-1}, prevStepMap);

        pair<int, int> step({gridSize-1,gridSize-1});
        while (prevStepMap.find(step) != prevStepMap.end()) {
            std::cout << "[" << step.first << "," << step.second << "] ";
            step = prevStepMap[step];
        }
        std::cout << std::endl;
    }
};

int main()
{
    vector<std::bitset<gridSize>> grid(gridSize);
    std::bitset<gridSize> row(0);
    for (int i = 0; i < gridSize; i++) {
        if (i % 3 != 0)
            row |= 1 << (rand() % gridSize);
        else
            row.reset();
        grid[i] = row;
    }

    // for (auto& row : grid) {
    //     string tmpStr = row.to_string();
    //     std::reverse(tmpStr.begin(), tmpStr.end());
    //     std::cout << tmpStr << std::endl;
    // }
    // std::cout << std::endl;
    
    Solution::maxPathSum(grid);

    return 0;
}