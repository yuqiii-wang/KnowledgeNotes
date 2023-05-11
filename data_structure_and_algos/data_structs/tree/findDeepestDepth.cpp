// find the deepest depth of a binary tree

#include "TreeNode.hpp"

class Solution {
public:
    int depth = 0;;
    int depthMaxCount = 0;
    void findMaxDepth(TreeNode* root) {

        if (!root) return ;

        depth++;
        depthMaxCount = std::max(depth, depthMaxCount);
        maxDepth(root->left); 
        maxDepth(root->right); 
        depth--;

    }
    int maxDepth(TreeNode* root) {
        findMaxDepth(root);
        return depthMaxCount;
    }
};