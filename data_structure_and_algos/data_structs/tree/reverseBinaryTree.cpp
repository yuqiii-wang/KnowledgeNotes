// input: root = [1,null,2,3]
// output: [1,3,2]
#include <vector>

#include "TreeNode.hpp"

class Solution {
public:
    static TreeNode* invertTree(TreeNode* root) {
        if (!root) return nullptr;

        TreeNode *leftNode = invertTree(root->left);
        TreeNode *rightNode = invertTree(root->right);

        root->left = rightNode;
        root->right = leftNode;

        return root;
    }
};

int main()
{
    std::vector<int> vec{3,2,1,2,3,4,5};
    TreeNode *root = buildBinaryTreeFromVec(vec);
    Solution::invertTree(root);
    return 0;
}