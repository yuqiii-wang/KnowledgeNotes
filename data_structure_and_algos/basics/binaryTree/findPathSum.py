"""
Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that adding up all the values along the path equals the given sum.
"""
import numpy as np
from binaryTreeConstrct import binaryTree

class findPathSum:

	def __init__(self, treeRoot, sumVal):
		self.tempSum = 0
		self.answer = self.__findPathSum(treeRoot, sumVal)

	def __findPathSum(self, Node, sumVal):
		if Node.root is None:
			return False
		else:
			self.tempSum += Node.root

		if self.tempSum==sumVal:
			return True

		tempBoolean = self.__findPathSum(Node.left, sumVal) or self.__findPathSum(Node.right, sumVal)

		self.tempSum -= Node.root

		return tempBoolean

	def __str__(self):
		return str(self.answer)


if __name__=="__main__":
	# set parameteres of a binary tree
	np.random.seed(111)
	val_range = 10
	size_range = 10
	array = list(range(size_range))
	print(array)
	tree = binaryTree(element=array, _Root_Start=True)
	print(findPathSum(tree, 6))