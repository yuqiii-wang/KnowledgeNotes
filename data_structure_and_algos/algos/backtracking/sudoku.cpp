// use bit to represent number's presence
// (011000100)_2 represent the number 3, 7 and 8 have shown up in a line/block

// bit operation (bitset): by complement, there is (~b)+1 = (-b), 
// where ~b is the bit flipping operation taking 0 -> 1 and 1 -< 0

#include <vector>
#include <bitset>

using namespace std;

class Solution {

private:
    vector<bitset<9>> rows;
    vector<bitset<9>> cols;
    vector<vector<bitset<9>>> cells;
public:
    bitset<9> getPossibleStatus(int x, int y)
    {
        return ~(rows[x] | cols[y] | cells[x / 3][y / 3]);
    }
};