// Definition for a Node.
function Node(val,children) {
    this.val = val;
    this.children = children;
}

var preorder = function(root) {
    if (root.length == 0){
        return [];
    }
    var stack = [root];
    var retArray = [];
    while (stack.length != 0){
        var tmpNode = stack.pop();
        retArray.push(tmpNode.val);
        if (tmpNode.children == null){
            continue;
        }
        tmpArr = tmpNode.children.reverse();
        for (var idx in tmpArr){
            stack.push(tmpArr[idx]);
        }
    }
    return retArray;
};

// execution enrty:
if (require.main === module){
    var ROOT = new Node(0, [
        new Node(1, [
            new Node(4, null),
            new Node(5, null)
        ]),
        new Node(2, [
            new Node(6, null),
            new Node(7, null)
        ]),
        new Node(3, [
            new Node(8, null)
        ]),
    ]);
    console.log(preorder(ROOT));
}