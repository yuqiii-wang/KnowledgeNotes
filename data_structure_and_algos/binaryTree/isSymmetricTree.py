"""
Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).
"""

import numpy as np
from binaryTreeConstrct import binaryTree

class isSymmetricTree:

	def __init__(self, treeRoot):
		if treeRoot.root is None:
			self.answer = "This is an empty tree"
		else:
			self.answer = self.__checkSymmetric(treeRoot.left, treeRoot.right)

	def __checkSymmetric(self, leftNode, rightNode):
		if leftNode.root is None or rightNode.root is None:
			return leftNode.root==rightNode.root
		if leftNode.root != rightNode.root:
			return False

		return self.__checkSymmetric(leftNode.left, rightNode.right) and self.__checkSymmetric(leftNode.right, rightNode.left)

	def __str__(self):
		return str(self.answer)


if __name__=="__main__":
	# set parameteres of a binary tree
	np.random.seed(111)
	val_range = 10
	size_range = 20
	array = [3,3,2,3,3,2,1] # this list is symmetric in binary tree form
	print(array)
	tree = binaryTree(element=array, _Root_Start=True)
	print(isSymmetricTree(tree))
	array = [3,3,2,3,3,4,1] # this list is not symmetric in binary tree form
	print(array)
	tree = binaryTree(element=array, _Root_Start=True)
	print(isSymmetricTree(tree))
	array = list(np.ones(1+2+4+8).astype('int64')) # this list is symmetric in binary tree form
	print(array)
	tree = binaryTree(element=array, _Root_Start=True)
	print(isSymmetricTree(tree))
