"""
Given a binary tree, return the level order 
traversal of its nodes' values. (ie, from left to right, level by level).
"""

import numpy as np

class binaryTree:

	def __init__(self, element, **kwargs):
		if kwargs['_Root_Start']:
			i = 0
			_count = 0
			depth = 0
			while(1):
				_count += 2**i
				i += 1
				depth += 1
				if _count > len(element):
					break
			_loop_count = depth
		else:
			_loop_count = kwargs['_loop_count']
			depth = kwargs['depth']

		if _loop_count > 0 and element:

			self.root = element.pop()

			_loop_count -= 1

			self.left = binaryTree(element, _Root_Start=False, depth=depth, _loop_count=_loop_count)
			self.right = binaryTree(element, _Root_Start=False, depth=depth, _loop_count=_loop_count)

		else:
			self.root = None
			_loop_count = depth

			return 

	def __str__(self):
		return str(self.root)

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
	size_range = 20
	array = list(np.random.randint(val_range, size=size_range))
	array = list(range(size_range))
	print(array)
	tree = binaryTree(element=array, _Root_Start=True)
	travsTreeByLevel_val = travsTreeByLevel(tree)
	print(travsTreeByLevel_val)