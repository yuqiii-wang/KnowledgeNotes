function TreeNode(val) {
    this.val = val;
    this.left = this.right = null;
}

function invertBinaryTree(root){
    if (root == null)
        return null;
    let leftNode = invertBinaryTree(root.left);
    let rightNode = invertBinaryTree(root.right);
    root.left = rightNode;
    root.right = leftNode;
    return root;
}

var treeRoot = new TreeNode(0);
treeRoot.left = new TreeNode(1);
treeRoot.right = new TreeNode(2);

console.log(treeRoot);
invertBinaryTree(treeRoot);
console.log(treeRoot);