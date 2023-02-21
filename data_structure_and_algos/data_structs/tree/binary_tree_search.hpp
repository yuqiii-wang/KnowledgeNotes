#include "node.hpp"

#pragma once

Node* searchBinaryTreeNode(Node* root, int searchVal)
{
    Node* temp = root;

    while (temp)
    {
        if (temp->data == searchVal)
            return temp;

        if (temp->left && temp->data < searchVal) {
            temp = temp->left;
        }
        else if (temp->right && temp->data > searchVal) {
            temp = temp->right;
        }
        else
            return nullptr;
    }

    return temp;
}
