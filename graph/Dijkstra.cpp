// https://www.techiedelight.com/find-shortest-path-in-maze/
// A maze is represented by 1 for "road" and 0 for "river",
// find the shortest path given a start point (x0, y0) and a destination point (x1, y1)

// Dijkstra cannot handle weighted routes
#include <vector>
#include <iostream>

using namespace std;

class dijkstra{
    vector<vector<int>> board;
    vector<vector<bool>> visitedRoute;
    vector<vector<int>> minRoute;
    int dest_x, dest_y;
    int size_x, size_y;
    int minDist, counter;

private:
    void do_dijkstra(int now_x, int now_y, int counter); // backtracking
    bool isSafe(int x, int y); // not encountering boarder
    bool isValid(int x, int y); // is '1' not '0'

public:
    dijkstra(vector<vector<int>> &board, int start_x,
             int start_y, int dest_x, int dest_y); // constructor
    ~dijkstra(); // destructor
    void printResult(); // show result (the shortest path)
    void printBoard(); // show the board
};

dijkstra::dijkstra(vector<vector<int>> &board_input, int start_x,
                   int start_y, int dest_x, int dest_y):board(board_input){

    this->size_x = static_cast<int>(board.size());
    this->size_y = static_cast<int>(board[0].size());
    this->dest_x = dest_x;
    this->dest_y = dest_y;

    vector<bool> visitedRow(size_y, false);
    vector<vector<bool>> visited(size_x, visitedRow);
    this->visitedRoute = visited;

    this->counter = 0;
    this->minDist = INT_MAX;

    do_dijkstra(start_x, start_y, 1);
}

void dijkstra::do_dijkstra(int x, int y, int counter){

    if (x == this->dest_x && y == this->dest_y){
        this->minDist = min(this->minDist, counter);
        return;
    }
    this->visitedRoute[x][y] = true;
    if (isSafe(x+1, y) && isValid(x+1, y))
        do_dijkstra(x+1, y, counter+1);
    if (isSafe(x, y+1) && isValid(x, y+1))
        do_dijkstra(x, y+1, counter+1);
    if (isSafe(x-1, y) && isValid(x-1, y))
        do_dijkstra(x-1, y, counter+1);
    if (isSafe(x, y-1) && isValid(x, y-1))
        do_dijkstra(x, y-1, counter+1);
    this->visitedRoute[x][y] = false;
    return;
}

bool dijkstra::isSafe(int x, int y){
    if (x >= 0 && y >= 0 && x < this->size_x && y < this->size_y)
        return true;
    return false;
}

bool dijkstra::isValid(int x, int y){
    if (this->visitedRoute[x][y] || board[x][y] == 0)
        return false;
    return true;
}

void dijkstra::printResult(){
    if (this->minDist == INT_MAX){
        cout << "Destination not reachable" << endl;
        return;
    }
    cout << "Min distance is " << minDist << endl;
}

void dijkstra::printBoard(){
    for (int i = 0; i < this->size_x; i++){
        for (int j = 0; j < this->size_y; j++)
            cout << board[i][j] << ' ';
        cout << endl;
    }
}

int main(){

    vector<int> row(5, 1);
    vector<vector<int>> board(5, row);
    board[1][1] = 0; board[1][2] = 0; board[2][2] = 0; board[3][0] = 0;

    dijkstra* D = new dijkstra(board, 0, 0, 4, 4);
    D->printBoard();
    D->printResult();

    return 0;
}