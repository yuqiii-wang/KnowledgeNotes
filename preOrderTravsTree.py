#post-order traversal of a binary tree

import numpy as np 

# construction of a tree

class binaryTree:

	def __init__(self, element, **kwargs):

		if '_Root_Start' not in kwargs:
			i = 0
			_count = 0
			self.depth = 0
			while(1):
				_count += 2**i
				i += 1
				self.depth += 1
				if _count > len(element):
					break

			self._loop_count = self.depth

		if '_loop_count' in kwargs:
			self._loop_count = kwargs['_loop_count']
		if 'depth' in kwargs:
			self.depth = kwargs['depth']

		if self._loop_count > 0 and element:

			self.root = element.pop()

			self._loop_count -= 1

			self.left = binaryTree(element, _Root_Start=False, depth=self.depth, _loop_count=self._loop_count)
			self.right = binaryTree(element, _Root_Start=False, depth=self.depth, _loop_count=self._loop_count)

		else:
			self.root = None
			self._loop_count = self.depth

			return 

	def __str__(self):
		return str(self.root)


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


