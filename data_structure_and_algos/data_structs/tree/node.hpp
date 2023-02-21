#include <iostream>
#include <list>

#pragma once

class Node{
public:
    int data;
    Node* left;
    Node* right;
    Node(int d){
        data=d;
        left=nullptr;
        right=nullptr;
    }
    Node() = default;
    ~Node() = default;
};
