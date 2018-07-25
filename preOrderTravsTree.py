#post-order traversal of a binary tree

import numpy as np 
from binaryTreeConstrct import binaryTree


class preOrderTravs:

	def __init__(self, tree):
		ret_array = []
		self.__Travs(tree, ret_array)

		self.array = ret_array


	def __Travs(self, node, ret_array):

		if node.root == None:
			return
		
		ret_array.insert(0, node.root)

		self.__Travs(node.left, ret_array)
		self.__Travs(node.right, ret_array)


	def __str__(self):
		return str(self.array)


if __name__=='__main__':
	# set parameteres of a binary tree
	np.random.seed(111)
	val_range = 10
	size_range = 10
	array = list(np.random.randint(val_range, size=size_range))
	print(array)
	tree = binaryTree(element=array)
	preOrderTravsObj = preOrderTravs(tree=tree)
	print(preOrderTravsObj)


