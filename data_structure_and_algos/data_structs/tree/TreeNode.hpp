#include <iostream>
#include <list>
#include <vector>

#pragma once

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

void insertTreeNode(TreeNode *root, int val){
    if (!root) {
        root = new TreeNode(val);
        return ;
    }

    if (root->val < val) {
        insertTreeNode(root->left, val);
    }
    else if (root->val > val) {
        insertTreeNode(root->right, val);
    }
    else {
        return;
    }
}

TreeNode *buildBinaryTreeFromVec(std::vector<int>& vec){

    if (vec.empty()) return nullptr;

    TreeNode *root = new TreeNode(vec[0]);
    for (int i = 1; i < vec.size(); i++) {
        insertTreeNode(root, vec[i]);
    }

    return root;
}
