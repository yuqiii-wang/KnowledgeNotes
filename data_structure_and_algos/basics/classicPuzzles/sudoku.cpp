#include <vector>
#include <iostream> 
#include <unordered_set> 

using namespace std;

// Given a board of size 9 by 9, add a number to the board and check whether the sudoku is valid
bool isValidSudoku(vector<vector<char>> &board, int value, int x, int y){
    board[x][y] = static_cast<char>(value + static_cast<int>('0'));

    // check subBox
    unordered_set<char> subBox; 
    int subBoxRow = static_cast<int>(x / 3);
    int subBoxCol = static_cast<int>(y / 3);
    for (int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            char tmpCell = board[subBoxRow * 3 + i][subBoxCol * 3 + j];
            if (subBox.find(tmpCell) == subBox.end())
                subBox.insert(tmpCell);
            else if (tmpCell == '.')
                continue;
            else 
                return false;
        }
    }

    // check row
    unordered_set<char> row; 
    for (int i = 0; i < 9; i++){
        char tmpCell = board[i][y];
        if (row.find(tmpCell) == row.end())
            row.insert(tmpCell);
        else if (tmpCell == '.')
            continue;
        else 
            return false;
    }

    // check col
    unordered_set<char> col; 
    for (int i = 0; i < 9; i++){
        char tmpCell = board[x][i];
        if (col.find(tmpCell) == col.end())
            col.insert(tmpCell);
        else if (tmpCell == '.')
            continue;
        else 
            return false;
    }
    return true;
}

void printBoard(vector<vector<char>> &board){
    for (int i = 0; i < 9; i++){
        for (int j = 0; j < 9; j++){
            cout << board[i][j] << ' ';
        }
        cout << endl;
    }
}

int main(){
    vector<char> row(9, '.');
    vector<vector<char>> board(9, row);
    board[1][2] = '2'; board[4][5] = '8'; board[8][1] = '6'; board[0][5] = '9';
    board[3][2] = '3'; board[3][8] = '1'; board[7][6] = '1'; board[1][5] = '5';
    board[3][5] = '6'; board[3][1] = '9'; board[4][3] = '7'; board[5][8] = '5';
    board[3][5] = '5'; board[8][5] = '4'; board[1][0] = '3'; board[2][3] = '1';
    board[6][5] = '2'; board[8][6] = '8'; board[1][6] = '7'; board[6][3] = '9';
    board[0][4] = '1'; board[0][7] = '3'; board[1][0] = '1'; board[7][3] = '5';

    bool flag = isValidSudoku(board, 1, 0, 0);

    printBoard(board);

    cout << '\n' << flag << endl;

    return 0;
}