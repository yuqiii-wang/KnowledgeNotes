import numpy as np

# The construction of a binary tree here shown below is via preOrder traversal method.
class binaryTree:

	def __init__(self, element, **kwargs):

		if kwargs['_Root_Start']:
			element = list(reversed(element))
			i = 0
			_count = 0
			depth = 0
			while(1):
				_count += 2**i
				i += 1
				depth += 1
				if _count >= len(element):
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