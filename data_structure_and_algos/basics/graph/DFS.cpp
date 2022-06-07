// Depth First Search
// https://www.geeksforgeeks.org/depth-first-search-or-dfs-for-a-graph/

#include <iostream>
#include <list>
#include <vector>

using namespace std;

// The key design principle is to append a list of adjacent nodes to each node
// to construct an adjacent matrix, by which we can iterate each of the node to 
// see whether they are all visited.
class Graph{
    int numV; // number of vertices
    list<int>* adjacent_mat;
    vector<bool> visited;

    private:
        void virtual DFS_do(int startNode); // internally run
    public:
        Graph(int numV); // constructor
        void addEdge(int parent, int child); // add edge between nodes
        void traverse(int startNode); // start traversing from a start node
        vector<int> graphVal;
};

Graph::Graph(int numV){
    this->numV = numV;
    adjacent_mat = new list<int>[numV];
    for(int i = 0; i < numV; i++)
        visited.push_back(false);
}

void Graph::addEdge(int parent, int child){
    adjacent_mat[parent].push_back(child); // to form an adjacency matrix
}

void Graph::traverse(int startNode){
    DFS_do(startNode);
}

void Graph::DFS_do(int node){
    visited[node] = true;
    graphVal.push_back(node);

    list<int>::iterator iter;
    for (iter = adjacent_mat[node].begin(); iter != adjacent_mat[node].end(); iter++)
        if (!visited[*iter])
            DFS_do(*iter);
}

int main(){
    Graph* G = new Graph(5);

    G->addEdge(2, 0);
    G->addEdge(2, 1);
    G->addEdge(2, 3);
    G->addEdge(0, 1);
    G->addEdge(1, 3);
    G->addEdge(1, 4);

    G->traverse(2);

    vector<int>::iterator iter_vec;
    for (iter_vec = G->graphVal.begin(); iter_vec != G->graphVal.end(); iter_vec++)
        cout << *iter_vec << '\t';
    cout << endl;
    
    return 0;
}