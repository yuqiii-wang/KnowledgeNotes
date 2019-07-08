# https://www.hackerrank.com/challenges/tree-huffman-decoding/problem

def decodeHuff(root, s):
	#Enter Your Code Here
    if root is None:
        return False
    resultList = []
    node = root
    for eachChar in list(s):
        node = node.left if eachChar == '0' else node.right
        if (node.left is None and node.right is None):
            resultList.append(node.data)
            node = root
    resultList = ''.join(resultList)
    print(resultList)
    return resultList