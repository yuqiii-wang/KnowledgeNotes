"""
Given a binary tree, count the number of uni-value subtrees.

A Uni-value subtree means all nodes of the subtree have the same value.
"""
import numpy as np
from binaryTreeConstrct import binaryTree

class findUniValSubTrees:

	def __init__(self, treeRoot):
		self.count = 0
		self.__findUniValSubTrees(treeRoot)

	def __findUniValSubTrees(self, Node):
		collected_elem = []

		if Node.root is None:
			return None 
		else:
			collected_elem.append(Node.root)

		leftArray = self.__findUniValSubTrees(Node.left)
		if leftArray is not None:
			for i in range(len(leftArray)):
				if leftArray[i] is not None:
					collected_elem.append(leftArray[i])
		rightArray = self.__findUniValSubTrees(Node.right)
		if rightArray is not None:
			for i in range(len(rightArray)):
				if rightArray[i] is not None:
					collected_elem.append(rightArray[i])

		if self.__ifUnique(collected_elem):
			self.count += 1

		return collected_elem

	def __ifUnique(self, array):
		return len(np.unique(array))==1

	def __str__(self):
		return str(self.count)


if __name__=="__main__":
	# set parameteres of a binary tree
	np.random.seed(111)
	val_range = 10
	size_range = 10
	array = list(np.ones(size_range).astype('int64'))
	print(array)
	tree = binaryTree(element=array, _Root_Start=True)
	print(findUniValSubTrees(tree))