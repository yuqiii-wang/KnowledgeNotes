# Definition for a Node.
class Node(object):
    def __init__(self, val, children):
        self.val = val
        self.children = children

class Solution(object):
    def isValidBinaryTree(self, root):
        if root is None:
            return False
        if root.children[0] is None or root.children[1] is None:
            return False

        leftBrnch = [root.children[0]]
        rightBrnch = [root.children[1]]
        while leftBrnch:
            tmpNode = leftBrnch.pop()
            if tmpNode.val > root.val:
                return False
            if tmpNode.children is None:
                continue                   
            leftBrnch.append(tmpNode.children)
        while rightBrnch:
            tmpNode = rightBrnch.pop()
            if tmpNode.val < root.val:
                return False
            if tmpNode.children is None:
                continue           
            rightBrnch.append(tmpNode.children)
        return True
            

if __name__ == "__main__":
    subNode_1 = Node(4, None)
    subNode_2 = Node(5, None)
    subNode_3 = Node(6, None)
    subNode_4 = Node(7, None)
    root = Node(0,[
            Node(1, [subNode_1, subNode_2]),
            Node(2, [subNode_3, subNode_4]),
            ])
    
    print(Solution().isValidBinaryTree(root))