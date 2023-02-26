// Breadth First Search

#include <bits/stdc++.h>
#include <vector>
#include <deque>
#include <list>
#include <iostream>

const int infty = INT_MAX;

using namespace std;
class Graph {
public:

    // No. of vertices
    int Vs; 
 
    // Adjacency/Edge mat
    vector<vector<int>> adj;
 
    Graph(int V); // Constructor
 
    // function to add an edge to graph
    // i and j are vertex index, w is edge cost
    void addEdge(int i, int j, int w);
};
 
Graph::Graph(int _V)
{
    this->Vs = _V;
    adj.resize(_V);
    for (auto& V : adj) {
        V.resize(_V);
        for (auto& v : V) {
            v = infty;
        }
    }
}
 
void Graph::addEdge(int i, int j, int w)
{
    adj[i][j] = w;
}

// traverse all vertices starting from vertex s
list<int> doBFS(Graph& g, int s)
{
    list<int> visitList;

    // Mark all the vertices as not visited
    vector<bool> visited;
    visited.resize(g.Vs, false);
 
    deque<int> queue;

    // Mark the current node as visited and enqueue it
    visited[s] = true;
    queue.push_back(s);
    visitList.push_back(s);
 
    while (!queue.empty()) {
        // Dequeue a vertex from queue
        // s is updated
        s = queue.front();
        queue.pop_front();
 
        // Get all vertices of the dequeued vertex s. 
        // If a vertex has not been visited,
        // then mark it visited and enqueue it
        for (int i = 0; i < g.Vs; i++) {
            if (g.adj[s][i] < infty) {
                if (!visited[i]) {
                    visited[i] = true;
                    queue.push_back(i);
                    visitList.push_back(i);
                }
            }
        }
    }

    return visitList;
}


// traverse all vertices starting from vertex s
list<int> doDFS(Graph& g, int s)
{
    list<int> visitList;

    // Mark all the vertices as not visited
    vector<bool> visited;
    visited.resize(g.Vs, false);
 
    deque<int> queue;

    // Mark the current node as visited and enqueue it
    queue.push_back(s);
 
    while (!queue.empty()) {
        // Dequeue a vertex from queue
        // s is updated
        s = queue.front();
        if (!visited[s])
            visitList.push_back(s);
        visited[s] = true;

        bool isAllVisited = true;
        for (int i = 0; i < g.Vs; i++) {
            if (g.adj[s][i] < infty) {
                isAllVisited &= (visited[i]); // if any vertex not yet visited, set to false
                if (!visited[i]) {
                    queue.push_front(i);
                    break;
                }
            }
        }
        if (isAllVisited)
            queue.pop_front();

    }

    return visitList;
}

int main()
{
    /*
                0
               / \
              1   2
             / \   \
            3   6   7
           / \     / \
          4   5   8   9
    */
    Graph g = Graph(10);
    g.addEdge(0, 1, 1);
    g.addEdge(0, 2, 1);
    g.addEdge(1, 3, 1);
    g.addEdge(1, 6, 1);
    g.addEdge(3, 4, 1);
    g.addEdge(3, 5, 1);
    g.addEdge(2, 7, 1);
    g.addEdge(7, 8, 1);
    g.addEdge(7, 9, 1);

    list<int> visitListBFS = doBFS(g, 0);
    cout << "BFS: ";
    for (auto& v : visitListBFS) {
        cout << v << " ";
    }
    cout << endl;

    list<int> visitListDFS = doDFS(g, 0);
    cout << "DFS: ";
    for (auto& v : visitListDFS) {
        cout << v << " ";
    }
    cout << endl;

    return 0;
}