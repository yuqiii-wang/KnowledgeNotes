#include "node.hpp"

#pragma once

Node* insertBinaryTreeNode(Node* root, Node* node) 
{
    if (root == nullptr)
        return node;

    Node* temp = root;

    while (temp)
    {
        if (temp->data < node->data)
        {
            if (temp->left != nullptr)
                temp = temp->left;
            else
                temp->left = node;
        }
        else if (temp->data > node->data)
        {
            if (temp->right != nullptr)
                temp = temp->right;
            else
                temp->right = node;
        }
        else { // equal, do nothing
            break;
        }
    }

    return root;
}

// to traverse all tree elements
void printBinaryTreeInOrder(Node* root)
{
    Node* temp = root;
    std::list<Node*> orderList;
    while (temp != nullptr || !orderList.empty()) {
        if (temp != nullptr) {
            orderList.push_front(temp);
            temp = temp->left;
        }
        else {
            temp = orderList.front();
            std::cout << temp->data << " ";
            orderList.pop_front();
            temp = temp->right;
        }
    }
    std::cout << std::endl;
}
