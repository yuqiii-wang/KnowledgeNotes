# Definition for a Node.
class Node(object):
    def __init__(self, val, children):
        self.val = val
        self.children = children

class Solution(object):
    def preorder(self, root):
        """
        :type root: Node
        :rtype: List[int]
        """
        if root is None:
            return []
        
        stack, output = [root, ], []            
        while stack:
            root = stack.pop()
            output.append(root.val)
            if root.children is None:
                continue
            stack.extend(root.children[::-1])
                
        return output

if __name__ == "__main__":
    subNode_1 = Node(4, None)
    subNode_2 = Node(5, None)
    subNode_3 = Node(6, None)
    subNode_4 = Node(7, None)
    subNode_5 = Node(8, None)
    subNode_6 = Node(9, None)
    root = Node(0,[
            Node(1, [subNode_1, subNode_2]),
            Node(2, [subNode_3, subNode_4]),
            Node(3, [subNode_5, subNode_6]),
            ])
    
    print(Solution().preorder(root))