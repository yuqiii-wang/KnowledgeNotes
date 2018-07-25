"""
Find maximum depth of a binary tree
"""

import numpy as np
from binaryTreeConstrct import binaryTree

class findMaxDepth:

	def __init__(self, treeRoot):

		self.max_num = self.__findMax(treeRoot)

	def __findMax(self, node):

		if node.root is None:
			return 0

		return 1+max(self.__findMax(node.left), self.__findMax(node.right))

	def __str__(self):
		return str(self.max_num)


if __name__=="__main__":
	# set parameteres of a binary tree
	np.random.seed(111)
	val_range = 10
	size_range = 20
	array = list(np.random.randint(val_range, size=size_range))
	array = list(range(size_range))
	print(array)
	tree = binaryTree(element=array, _Root_Start=True)
	print(findMaxDepth(tree))