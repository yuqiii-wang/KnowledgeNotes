// Breadth First Search


#include <iostream>
#include <vector>
#include <list>
#include <queue>

using namespace std;

class BFS{
private:
    void __BFS_do(int startNode); // internally run
public:
    int numV; // number of vertices
    list<int>* adjacent_mat;
    vector<bool> visited;
    BFS(int numV); // constructor
    void addEdge(int parent, int child); // add edge between nodes
    void traverse(int startNode); // start traversing from a start node
    void printNodes();
    queue<int> tmpGraphVal; // temperorily store graph val for iteration purposes
    queue<int> graphVal; // place traversed nodes on this stack
};

BFS::BFS(int numV){
    adjacent_mat = new list<int>[numV];
    for (int i = 0 ; i < numV; i++)
        visited.push_back(false);
}

void BFS::addEdge(int parent, int child){
    adjacent_mat[parent].push_back(child);
};

void BFS::traverse(int startNode){
    visited[startNode] = true;
    tmpGraphVal.push(startNode);
    __BFS_do(startNode);
};

void BFS::printNodes(){
    while(!graphVal.empty()){
        int nodeVal = graphVal.front();
        graphVal.pop();
        cout << nodeVal << endl;
    }
};

void BFS::__BFS_do(int startNode){
    list<int>::iterator iter;
    while(!tmpGraphVal.empty()){
        int nodeVal = tmpGraphVal.front();
        graphVal.push(nodeVal);
        tmpGraphVal.pop();
        for (iter = adjacent_mat[nodeVal].begin(); iter != adjacent_mat[nodeVal].end(); iter++){
            if (!visited[*iter]){
                visited[*iter] = true;
                tmpGraphVal.push(*iter);
            }
        }
    }
};

int main(){
    BFS* G = new BFS(5);
    
    G->addEdge(2, 0);
    G->addEdge(2, 1);
    G->addEdge(2, 3);
    G->addEdge(0, 1);
    G->addEdge(1, 3);
    G->addEdge(1, 4);
    
    G->traverse(2);
    
    G->printNodes();
    
    return 0;
}
