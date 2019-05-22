// Definition for a binary tree node.
function Node(val, left=null, right=null) {
    this.val = val;
    this.left = left;
    this.right = right;
}

function __travTreeLeft(node, rootVal){
    if (node == null) return true;
    if (node.val >= rootVal) return false;
    return __travTreeLeft(node.left, rootVal) & 
        __travTreeLeft(node.right, rootVal);
}

function __travTreeRight(node, rootVal){
    if (node == null) return true;
    if (node.val <= rootVal) return false;
    return __travTreeRight(node.left, rootVal) & 
        __travTreeRight(node.right, rootVal);
}

var isValidBinaryTree = function(root) {
    if (root == null) return true;
    stack = [];
    rootVal = root.val;
    return __travTreeLeft(root.left, rootVal) & 
        __travTreeRight(root.right, rootVal);
};

// execution enrty:
if (require.main === module){
    var ROOT = new Node(0, 
        this.left = new Node(1, 
            this.left = new Node(4, 
                this.left = null,
                this.right = null),
            this.right = new Node(5, 
                this.left = null,
                this.right = null)
        ),
        this.right = new Node(2, 
            this.left = new Node(6, 
                this.left = null,
                this.right = null),
            this.right = new Node(7, 
                this.left = null,
                this.right = null)
        )
    );
    console.log(ROOT);
    console.log(isValidBinaryTree(ROOT));
}