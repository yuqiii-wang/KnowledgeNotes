#include <iostream>
#include <bits/stdc++.h>
#include <vector>

const int INFTY = INT_MAX;
const int gridSize = 6;

void  dijkstra(std::vector<std::vector<int>>& e, 
        std::vector<int>& dis, 
        std::vector<int>& visited)
{
    int n = gridSize;

    // edge matrix init
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            if(i==j) 
                e[i][j]=0;
            else 
                e[i][j]=INFTY;

    // define reachable edges
    // refer to resources/dijkstra_directed_graph.png to see the graph
    // for example, `e[0][1]=10;` means that the cost from vertex 0 to vertex 1 is 10
    e[0][1]=10;
    e[0][5]=3;
    e[1][2]=7;
    e[1][3]=5;
    e[3][2]=4;
    e[3][4]=7;
    e[3][0]=3;
    e[5][1]=2;
    e[5][3]=6;
    e[5][4]=1;

    // init distances to vertex 0
    for(int i=0;i<n;i++)
        dis[i]=e[0][i];

    // only the vertex 0 is visited when started
    for(int i=0;i<n;i++)
        visited[i]=0;
    visited[0]=1;

    // core Dijkstra loop
    int u; // to record the closest vertex
    for(int i=0;i<n;i++)
    {
        // find the not yet visited v_j that is closest to vertex 0
        int min=INFTY;
        for(int j=0;j<n;j++)
        {
            if(visited[j]==0 && dis[j]<min)
            {
                min=dis[j];
                u=j;
            }
        }

        // from the new visited vertex u, check distances of u to all other vertices
        // if small, update distance from vertex 0 to u, plus from u to v.
        visited[u]=1; // mark the new vertex as visited
        for(int v=0;v<n;v++)
        {
            if(e[u][v] < INFTY) // prevent int overflow
            {
                if(dis[v] > dis[u]+e[u][v])
                    dis[v] = dis[u]+e[u][v];
            }
        }

    }
}

int main()
{
    std::vector<std::vector<int>> e(gridSize, std::vector<int>(gridSize, INFTY));
    std::vector<int> dis = std::vector<int>(gridSize, INFTY);
    std::vector<int> visited = std::vector<int>(gridSize, 0);

    dijkstra(e, dis, visited);
    for (int i = 0; i < gridSize; i++)
        std::cout << "From vertex 0 to vertex " << i 
        << " distance is " << dis[i] << std::endl;

}