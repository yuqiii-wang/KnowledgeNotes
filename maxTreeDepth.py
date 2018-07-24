"""
Find maximum depth of a binary tree
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