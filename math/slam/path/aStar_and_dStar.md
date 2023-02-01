# A* and D* Global Path Planning

### Spanning Tree

A spanning tree $T$ of an undirected graph $G$ is a subgraph that is a tree which includes **all** of the vertices of $G$.
An undirected graph $G$ can have many spanning trees.

Given the example below, all vertices are traversed/connected. Of course, there are many ways to traverse every vertex in this graph.

<div style="display: flex; justify-content: center;">
      <img src="imgs/spanning_tree_example.png" width="20%" height="20%" alt="spanning_tree_example">
</div>
</br>

## A Star 

A* is a global path planning algorithm with a known goal.