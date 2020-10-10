# https://www.hackerrank.com/challenges/binary-search-tree-lowest-common-ancestor/problem

def lca(root, v1, v2):
    #Enter your code here
    while(True):
        if (root.info > v1 and root.info > v2):
            root = root.left
        elif (root.info < v1 and root.info < v2):
            root = root.right
        else:
            return root