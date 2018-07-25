"""
Given a binary tree, return the level order 
traversal of its nodes' values. (ie, from left to right, level by level).
"""
import numpy as np
from binaryTreeConstrct import binaryTree

class travsTreeByLevel:

	def __init__(self, treeRoot):
		self.ret_array = []
		self.__travs(treeRoot)

	def __travs(self, treeRoot):
		i = 0
		nodePerLevel = [treeRoot]
		nodePerNextLevel = []

		self.ret_array.append([nodePerLevel[0].root])

		while(1):
			if not nodePerLevel:
				self.ret_array.pop()
				break

			self.ret_array.append([])

			num_elemPerLevel = len(nodePerLevel)

			for elemPerLevel in range(num_elemPerLevel):
				if nodePerLevel[elemPerLevel].left.root is not None:
					nodePerNextLevel.append(nodePerLevel[elemPerLevel].left)

				if nodePerLevel[elemPerLevel].right.root is not None:
					nodePerNextLevel.append(nodePerLevel[elemPerLevel].right)

			nodePerLevel = nodePerNextLevel
			nodePerNextLevel = []

			i += 1

			for elemPerLevel in range(len(nodePerLevel)):
				self.ret_array[i].append(nodePerLevel[elemPerLevel].root)

	def __str__(self):
		return str(self.ret_array)


if __name__=='__main__':
	# set parameteres of a binary tree
	np.random.seed(111)
	val_range = 10
	size_range = 10
	array = list(np.random.randint(val_range, size=size_range))
	array = list(range(size_range))
	print(array)
	tree = binaryTree(element=array, _Root_Start=True)
	travsTreeByLevel_val = travsTreeByLevel(tree)
	print(travsTreeByLevel_val)