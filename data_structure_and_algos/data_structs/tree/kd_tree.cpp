/*
    The k-d tree is a binary tree in which every node is a k-dimensional point. 

    If for a particular split the "x" axis is chosen, 
    all points in the subtree with a smaller "x" value than the node will appear in the left subtree, 
    and all points with a larger "x" value will be in the right subtree.
*/

#include <vector>
#include <iostream>

class Node
{
    int val;
};