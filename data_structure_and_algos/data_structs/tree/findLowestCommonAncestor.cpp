// find the lowest common ancestor
// if an ancestor sees diverged values of `p` and `q` different from `ancestor->val`, 
// this ancestor is the lowest common ancestor

#include <vector>

#include "TreeNode.hpp"

class Solution {
public:
    static TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        TreeNode* ancestor = root;
        while (true) {
            if (ancestor->val > p->val && ancestor->val > q->val) {
                ancestor = ancestor->left;
            }
            else if (ancestor->val < p->val && ancestor->val < q->val) {
                ancestor = ancestor->right;
            }
            else {
                break;
            }
        }
        return ancestor;
    }
};


int main()
{
    std::vector<int> vec{6,2,8,0,4,7,9,NULL,NULL,3,5};
    TreeNode *root = buildBinaryTreeFromVec(vec);
    TreeNode *p = new TreeNode(3);
    TreeNode *q = new TreeNode(5);
    Solution::lowestCommonAncestor(root, p, q);

    return 0;
}